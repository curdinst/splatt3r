import json
import os
import sys

import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb

# Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.scannetpp.scannetpp as scannetpp
import src.mast3r_src.mast3r.model as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import utils.lod_utils as lod_utils
import workspace
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.mast3r_slam_geometry import constrain_points_to_ray, depth_map_to_points
import torchvision

class MAST3RGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        # Save the config
        self.config = config

        # The encoder which we use to predict the 3D points and Gaussians,
        # trained as a modified MAST3R model. The model's configuration is
        # primarily defined by the pretrained checkpoint that we load, see
        # MASt3R's README.md
        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=False,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        self.encoder.requires_grad_(False)

        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(self.config["grad_gaussian_dpt"])
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(self.config["grad_gaussian_dpt"])
        self.encoder.downstream_head1.gaussian_dpt_256.dpt.requires_grad_(self.config["grad_gaussian_256_dpt"])
        self.encoder.downstream_head2.gaussian_dpt_256.dpt.requires_grad_(self.config["grad_gaussian_256_dpt"])
        self.encoder.downstream_head1.gaussian_dpt_128.dpt.requires_grad_(self.config["grad_gaussian_128_dpt"])
        self.encoder.downstream_head2.gaussian_dpt_128.dpt.requires_grad_(self.config["grad_gaussian_128_dpt"])
        self.encoder.downstream_head1.coarseness_classifier.requires_grad_(self.config["grad_coarseness_prediction"])
        self.encoder.downstream_head2.coarseness_classifier.requires_grad_(self.config["grad_coarseness_prediction"])

        self.used_gpu_mem = 0
        if self.config["train_head_only"]:
            self.encoder.downstream_head1.gaussian_dpt_256.dpt.requires_grad_(False)
            self.encoder.downstream_head2.gaussian_dpt_256.dpt.requires_grad_(False)
            self.encoder.downstream_head1.gaussian_dpt_256.dpt.head.requires_grad_(True)
            self.encoder.downstream_head1.gaussian_dpt_256.dpt.head.requires_grad_(True)
            self.encoder.downstream_head1.gaussian_dpt_256.dpt.act_postprocess.requires_grad_(True)
            self.encoder.downstream_head1.gaussian_dpt_256.dpt.act_postprocess.requires_grad_(True)

        print(f" resolution = {self.config['resolution']}, head_only = {self.config['train_head_only']} use_lod = {self.config['use_lod']}")
        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder_512 = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )
        self.decoder_256 = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )
        self.decoder_128 = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )
        self.decoder_coarseness = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

        self.benchmarker = benchmarker.Benchmarker()

        # Loss criteria
        if self.config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if self.config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        if self.config.train_coarseness_prediction:
            self.crossentropy_criterion = torch.nn.CrossEntropyLoss(reduction="none") # reduction='none'

        self.save_hyperparameters()

    def forward(self, view1, view2):

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1_512, pred1_256, pred1_128, coarseness1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2_512, pred2_256, pred2_128, coarseness2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)
        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2_512['pts3d_in_other_view'] = pred2_512.pop('pts3d')
        pred2_512['means_in_other_view'] = pred2_512.pop('means')

        pred1_512['covariances'] = geometry.build_covariance(pred1_512['scales'], pred1_512['rotations'])
        pred2_512['covariances'] = geometry.build_covariance(pred2_512['scales'], pred2_512['rotations'])

        pred1_256['covariances'] = geometry.build_covariance(pred1_256['scales'], pred1_256['rotations'])
        pred2_256['covariances'] = geometry.build_covariance(pred2_256['scales'], pred2_256['rotations'])

        pred1_128['covariances'] = geometry.build_covariance(pred1_128['scales'], pred1_128['rotations'])
        pred2_128['covariances'] = geometry.build_covariance(pred2_128['scales'], pred2_128['rotations'])

        # print(f"pred1_256['covariances']: {pred1_256['covariances'].shape}")
        # print(f"pred1_256['opacities']: {pred1_256['opacities'].shape}")
        # pred1_256['covariances'] = pred1_512['covariances'][:,::2,::2,:]
        # pred2_256['covariances'] = pred2_512['covariances'][:,::2,::2,:]
        # pred1_256['opacities'] = pred1_512['opacities'][:,::2,::2,:]
        # pred1_256['opacities'] = pred1_512['opacities'][:,::2,::2,:]            
        
        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1_512['sh'])
            new_sh2 = torch.zeros_like(pred2_512['sh'])
            # print(f"original image max: {view1['original_img'].max()}, min: {view1['original_img'].min()}")
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1_512['sh'] = pred1_512['sh'] + new_sh1
            pred2_512['sh'] = pred2_512['sh'] + new_sh2

            sh1_256 = (new_sh1[:,::2,::2,:] + new_sh1[:,1::2,::2,:] + new_sh1[:,::2,1::2,:] + new_sh1[:,1::2,1::2,:])/ 4.0
            sh2_256 = (new_sh2[:,::2,::2,:] + new_sh2[:,1::2,::2,:] + new_sh2[:,::2,1::2,:] + new_sh2[:,1::2,1::2,:])/ 4.0
            pred1_256['sh'] = sh1_256
            pred2_256['sh'] = sh2_256

            sh1_128 = (sh1_256[:,::2,::2,:] + sh1_256[:,1::2,::2,:] + sh1_256[:,::2,1::2,:] + sh1_256[:,1::2,1::2,:])/ 4.0
            sh2_128 = (sh2_256[:,::2,::2,:] + sh2_256[:,1::2,::2,:] + sh2_256[:,::2,1::2,:] + sh2_256[:,1::2,1::2,:])/ 4.0

            pred1_128['sh'] = pred1_128['sh'] + sh1_128
            pred2_128['sh'] = pred2_128['sh'] + sh2_128


        mean_0_mask = (pred1_512['means'][:, :, :, 0] == 0.0) & (pred1_512['means'][:, :, :, 1] == 0.0) & (pred1_512['means'][:, :, :, 2] == 0.0)
        # print(f"mean_0_mask: {mean_0_mask.shape}, {mean_0_mask.sum()}")
        if self.config["use_calib_and_gt_depth"]:
            # assert False, "constrain to ray with T_21"
            intrinsics1 = view1['camera_intrinsics']
            intrinsics2 = view2['camera_intrinsics']
            T_wc1 = view1['camera_pose'] # c2w
            T_wc2 = view2['camera_pose']
            # print(f"extinsics1: {T_wc1}")
            
            depth1 = view1['depthmap']
            depth2 = view2['depthmap']
            depth1_mask = (depth1 > 0.0)
            depth2_mask = (depth2 > 0.0)
            # print(f"depth1_mask.shape = {depth1_mask.shape}, depth2_mask.shape = {depth2_mask.shape}")
            depth1_mask_256 = depth1_mask[:,::2,::2] & depth1_mask[:,1::2,::2] & depth1_mask[:,::2,1::2] & depth1_mask[:,1::2,1::2]
            depth2_mask_256 = depth2_mask[:,::2,::2] & depth2_mask[:,1::2,::2] & depth2_mask[:,::2,1::2] & depth2_mask[:,1::2,1::2]
            depth1_mask_128 = depth1_mask_256[:,::2,::2] & depth1_mask_256[:,1::2,::2] & depth1_mask_256[:,::2,1::2] & depth1_mask_256[:,1::2,1::2]
            depth2_mask_128 = depth2_mask_256[:,::2,::2] & depth2_mask_256[:,1::2,::2] & depth2_mask_256[:,::2,1::2] & depth2_mask_256[:,1::2,1::2]
            # print(f"depth1 zero: {(depth1==0.0).sum()}, depth2 zero: {(depth2==0.0).sum()}")
            pred1_512['depth_mask'] = depth1_mask
            pred2_512['depth_mask'] = depth2_mask
            pred1_256['depth_mask'] = depth1_mask_256
            pred2_256['depth_mask'] = depth2_mask_256
            pred1_128['depth_mask'] = depth1_mask_128
            pred2_128['depth_mask'] = depth2_mask_128

            # depth1_mast3r = pred1_512['means'][..., -1]
            # zeromasks = (depth1_mast3r != 0.0) & (depth1 != 0.0)
            # ratio = (depth1_mast3r[zeromasks] / depth1[zeromasks]).mean()
            # print(f"depth1_mast3r: {depth1_mast3r.mean()}, depth1: {depth1.mean()}, ratio: {ratio}")

            T_c1_c2 = torch.linalg.inv(T_wc1) @ T_wc2 # T_wc1^-1 @ T_wc2 = T_c1c2
            # T_c1_c2[:3,3] = T_c1_c2[:3,3] * 0.1 # scale the translation by the ratio
            T_c2_c1 = torch.linalg.inv(T_c1_c2) # T_c1c2^-1 = T_c2c1
            
            means2_c1_hom = torch.cat([pred2_512['means_in_other_view'], torch.ones_like(pred2_512['means_in_other_view'][..., :1])], dim=-1) # (b, h, w, 4)
            means2_c2_hom = torch.einsum('bij, bhwj -> bhwi', T_c2_c1, means2_c1_hom) # (b, h, w, 4)
            depth2_mast3r = means2_c2_hom[..., 2] # (b, h, w)

            assert intrinsics1.shape[0] == 1, "assume single batch"
            assert intrinsics2.shape[0] == 1, "assume single batch"
            intrinsics1, intrinsics2 = intrinsics1[0,...], intrinsics2[0,...]
            h, w = view1['original_img'].shape[-2:]
            # print(f"h, w = {h}, {w}")
            # print(f"inrinsics__: {intrinsics1[0,0]}, {intrinsics1[1,1]}, {intrinsics1[0,2]}, {intrinsics1[1,2]}")

            # print(f"pred1_512[means].shape = {pred1_512['means'].shape}, pred2_512[means].shape = {pred2_512['means'].shape}")
            # print(f"depth1.shape = {depth1.shape}, depth2.shape = {depth2.shape}")
            # pred1_512['means'] = constrain_points_to_ray((h,w), pred1_512['means'], intrinsics1)
            # points2_homogeneous_c1 = torch.cat([pred2_512["means_in_other_view"], torch.ones_like(pred2_512["means_in_other_view"][..., :1])], dim=-1)
            # pred2_menas_c2 =  torch.einsum('bij, bhwj -> bhwi', T_c2_c1, points2_homogeneous_c1)
            # pred2_means_c2_ray = constrain_points_to_ray((h, w), pred2_menas_c2[..., :-1], intrinsics2)
            # pred2_means_c2_ray_homogeneous = torch.cat([pred2_means_c2_ray, torch.ones_like(pred2_means_c2_ray[..., :1])], dim=-1)
            # pred_means_ray_c1 = torch.einsum('bij, bhwj -> bhwi', T_c1_c2, pred2_means_c2_ray_homogeneous)
            # pred2_512['means_in_other_view'] = pred_means_ray_c1[..., :-1] # (b, h, w, 3)

            pred1_512['means'] = depth_map_to_points((h, w), depth1, intrinsics1)
            points2_c2 = depth_map_to_points((h, w), depth2, intrinsics2) # (b, h, w, 3)
            points2_homogeneous_c2 = torch.cat([points2_c2, torch.ones_like(points2_c2[..., :1])], dim=-1)

            points2_c1 = torch.einsum('bij, bhwj -> bhwi', T_c1_c2, points2_homogeneous_c2)
            pred2_512['means_in_other_view'] = points2_c1[..., :3] # (b, h, w, 3)

            pred1_512['pts3d'] = pred1_512['means']
            pred2_512['pts3d_in_other_view'] = pred2_512['means_in_other_view']

        # print(f"max sh1: {pred1_512['sh'].max()}, min sh1: {pred1_512['sh'].min()}")
        means1 = pred1_512['means']
        means2 = pred2_512['means_in_other_view']
        # print(f"means1 shape = {means1.shape}, means2 shape = {means2.shape}")
        pred1_256['means'] = (means1[:,::2,::2,:] + means1[:,1::2,::2,:] + means1[:,::2,1::2,:] + means1[:,1::2,1::2,:]) / 4.0
        pred2_256['means_in_other_view'] = (means2[:,::2,::2,:] + means2[:,1::2,::2,:] + means2[:,::2,1::2,:] + means2[:,1::2,1::2,:]) / 4.0
        
        means1_256 = pred1_256['means']
        means2_256 = pred2_256['means_in_other_view']

        pred1_128['means'] = (means1_256[:,::2,::2,:] + means1_256[:,1::2,::2,:] + means1_256[:,::2,1::2,:] + means1_256[:,1::2,1::2,:]) / 4.0
        pred2_128['means_in_other_view'] = (means2_256[:,::2,::2,:] + means2_256[:,1::2,::2,:] + means2_256[:,::2,1::2,:] + means2_256[:,1::2,1::2,:]) / 4.0

        if self.config.use_lod:
            # print(f"view1['original_img'].shape = {view1['original_img'].shape}")
            # print(f"pred1_512['pts3d'].shape = {pred1_512['pts3d'].shape}")
            H,W = view1['original_img'].shape[2:]
            # print(f"H,W = {H}, {W}")
            confidence_threshold = 1.5
            # print(f"pred1_512.keys() = {pred1_512.keys()}")
            valid = (pred1_512["conf"] > confidence_threshold)
            valid2 = (pred2_512["conf"] > confidence_threshold)
            device = "cuda:0"

            th_rgb = self.config.th_rgb
            th_depth = self.config.th_depth
            mask1_256 = torch.zeros((pred1_512["pts3d"].shape[0], H//2, W//2), dtype=torch.bool, device=device)
            mask2_256 = torch.zeros((pred2_512["pts3d"].shape[0], H//2, W//2), dtype=torch.bool, device=device)
            mask1 = torch.zeros((pred1_512["pts3d"].shape[0], H, W), dtype=torch.bool, device=device)
            mask2 = torch.zeros((pred2_512["pts3d"].shape[0], H, W), dtype=torch.bool, device=device)
            for b in range(pred1_512["pts3d"].shape[0]):
                # print(f"batch {b}, pred1_512['pts3d'].shape = {pred1_512['pts3d'].shape}, pred1_512['means'].shape = {pred1_512['means'].shape} valid.shape = {valid.shape}")
                mask1_256[b, ...], mask1[b, ...] =  lod_utils.get_mask(view1['original_img'][b, ...], pred1_512["pts3d"][b, ..., -1], valid[b, ...], device, H, W, th_depth=th_depth, th_rgb=th_rgb)
                
                # print(f"mask1_256.shape = {mask1_256.shape}, mask1.shape = {mask1.shape}")
                # print(f"mask1_256[b, ...].shape = {mask1_256[b, ...].shape}, mask1[b, ...].shape = {mask1[b, ...].shape}")

                mask2_256[b, ...] , mask2[b, ...] =  lod_utils.get_mask(view2['original_img'][b, ...], pred2_512["pts3d"][b, ..., -1], valid2[b, ...], device, H, W, th_depth=th_depth, th_rgb=th_rgb)
                
                # mask1_256[b, ...], mask1[b, ...] = lod_utils.get_mask(view1['original_img'][b, ...], pred1_512["pts3d"][b, ..., -1], valid, device, H, W, th_depth=th_depth, th_rgb=th_rgb)
                # mask2_256[b, ...], mask2[b, ...] = lod_utils.get_mask(view2['original_img'][b, ...], pred2_512["pts3d"][...,-1][b, ...], valid, device, H, W, th_depth=th_depth, th_rgb=th_rgb)
            pred1_combined = lod_utils.apply_mask_to_gaussians(pred1_512, pred1_256, mask1, mask1_256)
            pred2_combined = lod_utils.apply_mask_to_gaussians(pred2_512, pred2_256, mask2, mask2_256)
            pred2_combined['means_in_other_view'] = pred2_combined.pop('means')
            pred1_combined['mask_highres'] = mask1
            pred2_combined['mask_highres'] = mask2

        if self.config.coarseness_predictions:
            pred1_combined, pred2_combined = {}, {}
            predicted_classes = ((pred1_512, pred1_256, pred1_128, coarseness1, pred1_combined), (pred2_512, pred2_256, pred2_128, coarseness2, pred2_combined))
            for pred, pred_256, pred_128, coarseness, pred_combined in predicted_classes:
                classes = torch.argmax(coarseness, dim=1) # coarseness: (b, c, h, w) -> classes: (b, h, w)
                # print(f"calasses.shape: {classes.shape}, coarseness.shape: {coarseness.shape}")
                mask_512_use = (classes == 0) & pred['depth_mask']
                mask_256_use = (classes == 1) & pred['depth_mask']
                mask_128_use = (classes == 2) & pred['depth_mask']

                # Save mask_512_use as an image
                # torchvision.utils.save_image(mask_512_use.float(), f"mask_512_use.png")
                # torchvision.utils.save_image(mask_256_use.float(), f"mask_256_use.png")
                # torchvision.utils.save_image(mask_128_use.float(), f"mask_128_use.png")
                mask_256_xor = mask_256_use[:,::2,::2] ^ mask_256_use[:,1::2,::2] ^ mask_256_use[:,::2,1::2] ^ mask_256_use[:,1::2,1::2]
                mask_256_use_256 = mask_256_use[:,::2,::2] & mask_256_use[:,1::2,::2] & mask_256_use[:,::2,1::2] & mask_256_use[:,1::2,1::2]
                
                mask_256_xor_512 = torch.nn.functional.interpolate(mask_256_xor.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # print(f"mask_128_use.shape 1: {mask_128_use.shape}")
                mask_128_xor_256 = mask_128_use[:,::2,::2] ^ mask_128_use[:,1::2,::2] ^ mask_128_use[:,::2,1::2] ^ mask_128_use[:,1::2,1::2]
                mask_128_use_256 = mask_128_use[:,::2,::2] & mask_128_use[:,1::2,::2] & mask_128_use[:,::2,1::2] & mask_128_use[:,1::2,1::2]
                # print(f"mask_128_use.shape 2: {mask_128_use.shape}")
                mask_128_xor_128 = mask_128_use_256[:,::2,::2] ^ mask_128_use_256[:,1::2,::2] ^ mask_128_use_256[:,::2,1::2] ^ mask_128_use_256[:,1::2,1::2]
                mask_128_use_128 = mask_128_use_256[:,::2,::2] & mask_128_use_256[:,1::2,::2] & mask_128_use_256[:,::2,1::2] & mask_128_use_256[:,1::2,1::2]
                # print(f"mask_128_use.shape 3: {mask_128_use_256.shape}")
                # print(f"mask_128_xor_256.shape: {mask_128_xor_256.shape}, mask_128_xor_128.shape: {mask_128_xor_128.shape}, mask_256_xor_512.shape: {mask_256_xor_512.shape}")
                # mask_128_xor_256_512 = torch.nn.functional.interpolate(mask_128_xor_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # mask_128_xor_128_256 = torch.nn.functional.interpolate(mask_128_xor_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # print(f"mask_128_xor_128_256.shape: {mask_128_xor_128_256.shape}, mask_128_xor_256_512.shape: {mask_128_xor_256_512.shape}, mask_256_xor_512.shape: {mask_256_xor_512.shape}")


                # mask_128_use_128 to use on 128 resolution
                mask_128_use_128_upsampled_256 = torch.nn.functional.interpolate(mask_128_use_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                mask_128_to_256 = mask_128_use_256 & ~mask_128_use_128_upsampled_256
                mask_256_use_256 = mask_256_use_256 | mask_128_to_256

                mask_128_use_256_upsampled_512 = torch.nn.functional.interpolate(mask_128_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                mask_128_to_512 = mask_128_use & ~mask_128_use_256_upsampled_512

                mask_128_upsampled_256 = torch.nn.functional.interpolate(mask_128_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # mask_256_use_256 = mask_256_use_256 | mask_128_xor_128_256
                mask_256_upsampled_512 = torch.nn.functional.interpolate(mask_256_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                mask_256_to_512 = mask_256_use & ~mask_256_upsampled_512
                mask_512_use = mask_512_use | mask_256_to_512 | mask_128_to_512

                # torchvision.utils.save_image(mask_512_use.float(), f"mask_512_use_1.png")
                # torchvision.utils.save_image(mask_256_use_256.float(), f"mask_256_use_256_1.png")
                # torchvision.utils.save_image(mask_128_use.float(), f"mask_128_use_1.png")

                # mask_256_use_upsampled = torch.nn.functional.interpolate(mask_256_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # mask_128_use_upsampled = torch.nn.functional.interpolate(mask_128_use_128.float().unsqueeze(1), scale_factor=4, mode='nearest').squeeze(1).bool()
                # torchvision.utils.save_image(mask_256_use_upsampled.float(), f"mask_256_use_upsampled.png")
                # torchvision.utils.save_image(mask_128_use_upsampled.float(), f"mask_128_use_upsampled.png")

                # colorimg = torch.stack([mask_512_use.float(), mask_256_use_upsampled.float(), mask_128_use_upsampled.float()], dim=1) # (3, h, w)
                # torchvision.utils.save_image(colorimg, f"mask_combined.png")

                

                # mask_256 = classes[:,::2,::2] & classes[:,1::2,::2] & classes[:,::2,1::2] & classes[:,1::2,1::2]
                # mask_128 = mask_256[:,::2,::2] & mask_256[:,1::2,::2] & mask_256[:,::2,1::2] & mask_256[:,1::2,1::2]
                # mask_128_upsampled_256 =  torch.nn.functional.interpolate(mask_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # mask_256 = mask_256 & ~mask_128_upsampled_256
                # mask_256_upsampled_512 = torch.nn.functional.interpolate(mask_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                # mask_512 = ~ mask_256_upsampled_512

                # mask_512_use = (mask_512 == 0)
                # mask_256_use = (mask_256 == 1)
                # mask_128_use = (mask_128 == 2)
                # print(f"mask_128_use.shape: {mask_128_use.shape}, mask_256_use.shape: {mask_256_use.shape}, mask_512_use.shape: {mask_512_use.shape}")
                # print(f"pred_128['means'].shape: {pred_128['means'].shape}, pred_256['means'].shape: {pred_256['means'].shape}, pred_512['means'].shape: {pred['means'].shape}")
                # print(f"mask_512.shape: {mask_512.shape}, mask_256.shape: {mask_256.shape}, mask_128.shape: {mask_128.shape}")
                for key in pred.keys():
                    if key not in ["means", "means_in_other_view", "opacities", "sh", "rotations", "scales", "covariances"]:
                        continue
                    tensor_list = []
                    # for b in range(pred["means"].shape[0]):
                    #     # print(f"fusing key, {key}, pred[key].shape: {pred[key].shape}, pred_lowres[key].shape: {pred_lowres[key].shape}")
                    #     tensor_list.append(torch.cat([pred[key][b,mask_512_use[b,...],...], 
                    #                                   pred_256[key][b,mask_256_use[b,...],...], 
                    #                                   pred_128[key][b,mask_128_use[b,...],...]]
                    #                                   , dim=0))
                    # pred_combined[key] = torch.stack(tensor_list, dim=0)
                    b=0
                    pred_combined[key] = torch.cat([pred[key][b,mask_512_use[b,...],...], 
                                                      pred_256[key][b,mask_256_use_256[b,...],...], 
                                                      pred_128[key][b,mask_128_use_128[b,...],...]]
                                                      , dim=0)
                    pred_combined[key] = pred_combined[key].unsqueeze(0) # add batch dimension
            # pred2_combined['means_in_other_view'] = pred2_combined.pop('means')
            # pred2_combined['pts3d_in_other_view'] = pred2_combined.pop('pts3d')

        # Update the keys to make clear that pts3d and means are in view1's frame
        # pred2_512['pts3d_in_other_view'] = pred2_512.pop('pts3d')
        # pred2_512['means_in_other_view'] = pred2_512.pop('means')
        # for key in pred2_combined.keys():
            # print(f"key. {key}, pred2_combined[key].shape = {pred2_combined[key].shape}")
        if self.config.train_coarseness_prediction:
            return pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2
        if self.config.use_lod or self.config.coarseness_predictions:
            # return pred1_512, pred2_512, pred1_256, pred2_256, pred1_combined, pred2_combined
            return pred1_combined, pred2_combined
        else:
            return pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128
        
    def training_step(self, batch, batch_idx):
        # device = torch.device('cuda:0')
        # free, total = torch.cuda.mem_get_info(device)
        # mem_used_MB = (total - free) / 1024 ** 2
        # print("mem_used_MB:", mem_used_MB)
        num_targets = len(batch['target'])
        # print(f"Training step {batch_idx}, len batch context {len(batch['context'])}, num targets: {num_targets}")
        print(f"Training scene {batch['scene']}, step {batch_idx}")
        # print(f"Training step {batch_idx}, batch size: {len(batch['context'])}")
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        if self.config.train_coarseness_prediction and not self.config.train_coarseness_prediction_img_grad:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            # for key in pred1_512.keys():
            #     if "pts3d" in key: continue
            #     pred1_512[key].to(self.device2)
            #     pred1_256[key].to(self.device2)
            #     pred1_128[key].to(self.device2)
            # for key in pred2_512.keys():
            #     if "pts3d" in key: continue
            #     pred2_512[key].to(self.device2)
            #     pred2_256[key].to(self.device2)
            #     pred2_128[key].to(self.device2)
            # coarseness1.to(self.device2)
            # coarseness2.to(self.device2)
            

            with torch.no_grad():
                color_512, _ = self.decoder_512(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
                color_256, _ = self.decoder_256(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                color_128, _ = self.decoder_128(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)



            free, total = torch.cuda.mem_get_info(self.device0)
            mem_used_MB = (total - free) / 1024 ** 2
            if mem_used_MB > 11000:
                print(f"Memory usage is too high: {mem_used_MB} MB, clearing cache")
                torch.cuda.empty_cache()
            # color_512.to(self.device1)
            # color_256.to(self.device1)
            # color_512.to(self.device1)
                # color_256, _ = self.decoder_256(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                # color_128, _ = self.decoder_128(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
            # free, total = torch.cuda.mem_get_info(device)
            # mem_used_MB = (total - free) / 1024 ** 2
            # print("mem_used_MB 1:", mem_used_MB)
                
            pred1_coarseness, pred2_coarseness = self.create_coarseness_gaussians(pred1_512, pred2_512, coarseness1, coarseness2)

            # print(f"pred1_coarseness['means]: {torch.min(torch.abs(pred1_coarseness['means'].mean(dim=-1)))}")
            # print(f"pred1_coarseness[scales] {pred1_coarseness['scales'].shape}, cov: {pred1_coarseness['covariances'].shape}")
            # argmax = torch.argmax(pred1_coarseness['scales'], keepdim=True)
            # mean_0_mask = (pred1_coarseness['means'][:, :, :, 0] == 0.0) & (pred1_coarseness['means'][:, :, :, 1] == 0.0) & (pred1_coarseness['means'][:, :, :, 2] == 0.0)
            # print(f"mean_0_mask: {mean_0_mask.shape}, {mean_0_mask.sum()}")
            # print(f"pred_1_coarseness[means] {pred1_coarseness['means'].shape}, {torch.abs(torch.min())}")
            # print(f"pred1_coarseness[scales] {argmax}")
            # print(f"pred_coarseness1 {pred1_coarseness['means'][argmax]}")
            # print(f"pred1_coarseness[scales] {pred1_coarseness['covariances'].max()}")
            # if batch_idx == 32 or batch_idx == 1:
            #     export.save_as_ply(pred1_coarseness, pred2_coarseness, "/home/curdinst/repos/splatt3r/results/coarsenesspred.ply")

            # free, total = torch.cuda.mem_get_info(device)
            # mem_used_MB = (total - free) / 1024 ** 2
            # print("mem_used_MB: 2", mem_used_MB)
            # print(f"pred1_coarseness[sh].grad_fn = {pred1_coarseness['sh'].grad_fn}")
            # print(f"pred1_coarseness[scales].mean() = {pred1_coarseness['scales'].mean()}")

            # if batch_idx == 299 or batch_idx == 300 or batch_idx == 1:
            #     for key in pred1_coarseness:
            #         print(f"pred1_coarseness[{key}].shape = {pred1_coarseness[key].shape}")
            #         print(f"pred1_coarseness[{key}].mean() = {pred1_coarseness[key].mean()}")
            #         print(f"pred1_coarseness[{key}].grad_fn = {pred1_coarseness[key].grad_fn}")
            #         print(f"pred1_coarseness[{key}].grad = {pred1_coarseness[key].grad}")
            #     for key in pred2_coarseness:
            #         print(f"pred2_coarseness[{key}].shape = {pred2_coarseness[key].shape}")
            #         print(f"pred2_coarseness[{key}].mean() = {pred2_coarseness[key].mean()}")
            #         print(f"pred2_coarseness[{key}].grad_fn = {pred2_coarseness[key].grad_fn}")
            #         print(f"pred2_coarseness[{key}].grad = {pred2_coarseness[key].grad}")
            coarseness_image, _ = self.decoder_coarseness(batch, pred1_coarseness, pred2_coarseness, (h, w)) # gets (b, v, c, h, w)
            # pred1_keys = list(pred1_512.keys()).copy()
            # pred2_keys = list(pred2_512.keys()).copy()            
            # for key in pred1_keys:
            #     if "pts3d" in key: continue
            #     del pred1_512[key], pred1_256[key], pred1_128[key]
            # for key in pred2_keys:
            #     if "pts3d" in key: continue
            #     del pred2_512[key], pred2_256[key], pred2_128[key]
            # pred1_keys = list(pred1_coarseness.keys()).copy()
            # pred2_keys = list(pred2_coarseness.keys()).copy()
            # for key in pred1_keys:
            #     if "pts3d" in key: continue
            #     del pred1_coarseness[key]
            # for key in pred2_keys:
            #     if "pts3d" in key: continue
            #     del pred2_coarseness[key]

            # coarseness_image = color_512 #self.decoder_512(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
            # del pred1_coarseness, pred2_coarseness
            # coarseness_image, _ = self.decoder(batch, pred1_512, pred2_512, (h, w))
            # free, total = torch.cuda.mem_get_info(device)
            # mem_used_MB = (total - free) / 1024 ** 2
            # print("mem_used_MB 3:", mem_used_MB)
            
            # print(f"coarseness_image.shape = {coarseness_image.shape}, pred1_coarseness['sh'].shape = {pred1_coarseness['sh'].shape}, pred2_coarseness['sh'].shape = {pred2_coarseness['sh'].shape}")
            mask = loss_mask.calculate_loss_mask(batch)
            # print(f"coarseness_image: {coarseness_image}")
            # free, total = torch.cuda.mem_get_info(device)
            # mem_used_MB = (total - free) / 1024 ** 2
            # print("mem_used_MB 4:", mem_used_MB)
            loss, num_gaussians = self.calculate_loss_3stage(batch, view1, view2,
                                       (color_512, color_256, color_128),
                                       coarseness_image,
                                        mask)
            # free, total = torch.cuda.mem_get_info(device)
            # mem_used_MB = (total - free) / 1024 ** 2
            # print("mem_used_MB 5:", mem_used_MB)
            del color_512, color_256, color_128, coarseness_image
            del pred1_coarseness, pred2_coarseness, coarseness1, coarseness2
            self.log_metrics('train', loss, mse=0, lpips=0, num_gaussians=num_gaussians, train_coarseness_prediction=True)
            # del pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2
            
            # free, total = torch.cuda.mem_get_info(device)
            # mem_used_MB = (total - free) / 1024 ** 2
            # print("mem_used_MB 6:", mem_used_MB)
            return loss
        
        elif self.config.train_coarseness_prediction_img_grad:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            loss = self.calculate_loss_coarseness_img_freq(view1, view2, pred1_512, pred2_512, coarseness1, coarseness2)
            self.log_metrics('train', loss, mse=0, lpips=0, num_gaussians=0, train_coarseness_prediction=True)
            return loss
        
        elif self.config.train_3stage_gaussian_heads:
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128 = self.forward(view1, view2)

            color_512, _ = self.decoder_512(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
            color_256, _ = self.decoder_256(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
            color_128, _ = self.decoder_128(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss_512, mse_512, lpips_512, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_512, pred2_512, color_512, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
            loss_256, mse_256, lpips_256, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color_256, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
            loss_128, mse_128, lpips_128, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_128, pred2_128, color_128, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
            loss = (loss_512, loss_256, loss_128)
            mse = (mse_512, mse_256, mse_128)
            lpips = (lpips_512, lpips_256, lpips_128)
            self.log_metrics('train', loss, mse, lpips, num_gaussians=num_gaussians)
            loss = (loss_512 + loss_256 + loss_128) / 3.0
            return loss
        
        elif self.config.use_lod or self.config.coarseness_predictions:
            # return pred1, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
            pred1_combined, pred2_combined = self.forward(view1, view2)

            color,_ = self.decoder_512(batch, pred1_combined, pred2_combined, (h, w), fused_gaussians=True) # gets (b, v, c, h, w)
            mask = loss_mask.calculate_loss_mask(batch)

            loss, mse, lpips, num_gaussians = self.calculate_loss(batch, view1, view2, pred1_combined, pred2_combined, color, mask)
            self.log_metrics('train', loss, mse, lpips, num_gaussians=num_gaussians)
            return loss

        # elif self.config.use_lod or self.config.coarseness_predictions:
        #     # return pred1_512, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
        #     pred1_combined, pred2_combined = self.forward(view1, view2)
        # else:
        #     pred1, pred2, pred1, pred2, pred1_128, pred2_128 = self.forward(view1, view2)
        
        # if self.config.resolution < 500:
        #     color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w))
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1_256, pred2_256, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=False
        #     )
        # elif self.config.use_lod:
        #     color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w), fused_gaussians=True)
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1_256, pred2_256, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=False
        #     )
        # else:
        #     color, _ = self.decoder(batch, pred1, pred2, (h, w))
        
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1, pred2, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=False
        #     )

        # # Log losses
        # self.log_metrics('train', loss, mse, lpips, num_gaussians=num_gaussians)
        # return loss

    def validation_step(self, batch, batch_idx):
        num_targets = len(batch['target'])
        # print(f"Validation scene {batch['scene']}, len batch context {len(batch['context'])}, {batch['context'][0].keys()} num targets: {num_targets}")
        print(f"validation step {batch['scene']}")
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        
        # Predict using the encoder/decoder and calculate the loss
        if self.config.train_coarseness_prediction and not config.train_coarseness_prediction_img_grad:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            with torch.no_grad():
                color_512, _ = self.decoder_512(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
                color_256, _ = self.decoder_256(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                color_128, _ = self.decoder_128(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
                # print(f"color_512.shape = {color_512.shape}, color_256.shape = {color_256.shape}, color_128.shape = {color_128.shape}")
            pred1_coarseness, pred2_coarseness = self.create_coarseness_gaussians(pred1_512, pred2_512, coarseness1, coarseness2)
            
            with torch.no_grad():
                coarseness_image, _ = self.decoder_coarseness(batch, pred1_coarseness, pred2_coarseness, (h, w)) # gets (b, v, c, h, w)
            # print(f"coarseness_image: {coarseness_image}")
            # print(f"color_512: {color_512}")
            # print(f"coarseness_image.shape = {coarseness_image.shape}, pred1_coarseness['sh'].shape = {pred1_coarseness['sh'].shape}, pred2_coarseness['sh'].shape = {pred2_coarseness['sh'].shape}")
            mask = loss_mask.calculate_loss_mask(batch)
            loss, num_gaussians = self.calculate_loss_3stage(batch, view1, view2,
                                       (color_512, color_256, color_128),
                                       coarseness_image,
                                       mask)
             # Delete the intermediate tensors that are no longer needed
            del color_512, color_256, color_128, coarseness_image
            del pred1_coarseness, pred2_coarseness, coarseness1, coarseness2
            self.log_metrics('val', loss, mse=0, lpips=0, num_gaussians=num_gaussians, train_coarseness_prediction=True)
            return loss
        elif self.config.train_coarseness_prediction_img_grad:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            loss = self.calculate_loss_coarseness_img_freq(view1, view2, pred1_512, pred2_512, coarseness1, coarseness2)
            self.log_metrics('val', loss=loss, mse=0, lpips=0, num_gaussians=0, train_coarseness_prediction=True)
            return loss
        elif self.config.train_3stage_gaussian_heads:
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128 = self.forward(view1, view2)

            color_512, _ = self.decoder_512(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
            color_256, _ = self.decoder_256(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
            color_128, _ = self.decoder_128(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss_512, mse_512, lpips_512, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_512, pred2_512, color_512, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
            loss_256, mse_256, lpips_256, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color_256, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
            loss_128, mse_128, lpips_128, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_128, pred2_128, color_128, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
            loss = (loss_512, loss_256, loss_128)
            mse = (mse_512, mse_256, mse_128)
            lpips = (lpips_512, lpips_256, lpips_128)
            self.log_metrics('val', loss, mse, lpips, num_gaussians=num_gaussians)
            loss = (loss_512 + loss_256 + loss_128) / 3.0
            return loss
        elif self.config.use_lod or self.config.coarseness_predictions:
            # return pred1, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
            pred1_combined, pred2_combined = self.forward(view1, view2)

            color,_ = self.decoder_512(batch, pred1_combined, pred2_combined, (h, w), fused_gaussians=True) # gets (b, v, c, h, w)
            mask = loss_mask.calculate_loss_mask(batch)

            loss, mse, lpips, num_gaussians = self.calculate_loss(batch, view1, view2, pred1_combined, pred2_combined, color, mask)
            num_gaussians = pred1_combined['means'].shape[1] + pred2_combined['means_in_other_view'].shape[1]
            # print(f"num_gaussians: {num_gaussians}")
            self.log_metrics('val', loss, mse, lpips, num_gaussians=num_gaussians)
            return loss
            # loss = self
        # else:
        #     pred1, pred2, pred1, pred2, pred1_128, pred2_128 = self.forward(view1, view2)
        
        # if self.config.resolution < 500:
        #     color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w))
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1_256, pred2_256, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=False
        #     )
        # elif self.config.use_lod:
        #     color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w), fused_gaussians=True)
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1_256, pred2_256, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=False
        #     )
        # else:
        #     color, _ = self.decoder(batch, pred1, pred2, (h, w))
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1, pred2, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=False
        #     )

        # # print(f"Loss: {loss.item()}, MSE: {mse.item()}, LPIPS: {lpips.item()} ------------------------------------")
        # # Log losses
        # self.log_metrics('val', loss, mse, lpips, num_gaussians=num_gaussians)
        # return loss

    def test_step(self, batch, batch_idx):
        print(f"test step {batch_idx}, scene {batch['scene']}")
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        num_targets = len(batch['target'])

        if self.config.train_coarseness_prediction and not config.train_coarseness_prediction_img_grad:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            with torch.no_grad():
                color_512, _ = self.decoder(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
                color_256, _ = self.decoder(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                color_128, _ = self.decoder(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
                # print(f"color_512.shape = {color_512.shape}, color_256.shape = {color_256.shape}, color_128.shape = {color_128.shape}")
            pred1_coarseness, pred2_coarseness = self.create_coarseness_gaussians(pred1_512, pred2_512, coarseness1, coarseness2)
            with torch.no_grad():
                coarseness_image, _ = self.decoder(batch, pred1_coarseness, pred2_coarseness, (h, w)) # gets (b, v, c, h, w)
            # print(f"coarseness_image: {coarseness_image}")
            # print(f"color_512: {color_512}")
            print(f"coarseness_image.shape = {coarseness_image.shape}, pred1_coarseness['sh'].shape = {pred1_coarseness['sh'].shape}, pred2_coarseness['sh'].shape = {pred2_coarseness['sh'].shape}")
            mask = loss_mask.calculate_loss_mask(batch)
            loss, num_gaussians = self.calculate_loss_3stage(batch, view1, view2,
                                       (color_512, color_256, color_128),
                                       coarseness_image,
                                        mask)

            self.log_metrics('test', loss, mse=0, lpips=0, num_gaussians=num_gaussians, train_coarseness_prediction=True)
            return loss
        elif self.config.train_coarseness_prediction_img_grad:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            loss = self.calculate_loss_coarseness_img_freq(view1, view2, pred1_512, pred2_512, coarseness1, coarseness2)
            self.log_metrics('test', loss=loss, mse=0, lpips=0, num_gaussians=0, train_coarseness_prediction=True)
            return loss
        
        elif self.config.use_lod or self.config.coarseness_predictions:
            # return pred1, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
            pred1_combined, pred2_combined = self.forward(view1, view2)

            color,_ = self.decoder_512(batch, pred1_combined, pred2_combined, (h, w), fused_gaussians=True) # gets (b, v, c, h, w)
            mask = loss_mask.calculate_loss_mask(batch)

            loss, mse, lpips, num_gaussians = self.calculate_loss(batch, view1, view2, pred1_combined, pred2_combined, color, mask)
            num_gaussians = pred1_combined['means'].shape[1] + pred2_combined['means_in_other_view'].shape[1]
            # print(f"num_gaussians: {num_gaussians}")
            self.log_metrics('test', loss, mse, lpips, num_gaussians=num_gaussians)
            return loss
        # # Predict using the encoder/decoder and calculate the loss
        # with self.benchmarker.time("encoder"):
        #     pred1, pred2, pred1_256, pred2_256 = self.forward(view1, view2)
        # with self.benchmarker.time("decoder", num_calls=num_targets):
        #     if self.config.resolution < 500:
        #         color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w))
        #     elif self.config.use_lod:
        #         color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w), fused_gaussians=True)
        #     else:
        #         color, _ = self.decoder(batch, pred1, pred2, (h, w))
        
        # if self.config.resolution < 500 or self.config.use_lod:
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, ssim, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1_256, pred2_256, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=True
        #     )
        # else:
        #     # Calculate losses
        #     mask = loss_mask.calculate_loss_mask(batch)
        #     loss, mse, lpips, ssim, num_gaussians = self.calculate_loss(
        #         batch, view1, view2, pred1, pred2, color, mask,
        #         apply_mask=self.config.loss.apply_mask,
        #         average_over_mask=self.config.loss.average_over_mask,
        #         calculate_ssim=True
        #     )

        # # Log losses
        # self.log_metrics('test', loss, mse, lpips, ssim=ssim, num_gaussians=num_gaussians)
        # return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss_3stage(self, batch, view1, view2, colors, coarsenesses_image, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False, get_gt_coarseness=False):
        """
            mask: (b, v, h, w) - boolean mask indicating valid pixels
        """

        # print(f"len batch['target'] = {len(batch['target'])}")
        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        (predicted_color_512, predicted_color_256, predicted_color_128) = colors
        
        # print(f"predicted_color_512.shape: {predicted_color_512.shape}, target_color.shape: {target_color.shape}")
        # if self.config.use_lod:
        #     num_gaussians = pred1['sh'].shape[1] + pred2['sh'].shape[1]
        # else:
        #     num_gaussians = pred1['sh'].shape[1]*pred1['sh'].shape[2] + pred2['sh'].shape[1]*pred2['sh'].shape[2]
        coarseness_image_empty = coarsenesses_image.mean(dim=2) # (b, v, h, w)
        # mask_coarseness_render = torch.ones_like(coarseness_image_empty)
        # mask_coarseness_render[coarseness_image_empty == 0] = 0 # (b, v, h, w)
        # mask = mask * mask_coarseness_render
        # print(f"mask shape: {mask.shape}, mask.sum(): {mask.sum()}")
        if self.config.loss.apply_mask:
            if mask.sum() < 1:
                print(f"Skipping batch due to no valid pixels in the mask! batch['scene_id'] = {batch['scene']}")
                zero_loss = torch.tensor(0.0, device=target_color.device, requires_grad=True)
                if calculate_ssim:
                    return zero_loss, zero_loss
                return zero_loss, zero_loss
            target_color = target_color * mask[..., None, :, :]
            predicted_color_512 = predicted_color_512 * mask[..., None, :, :] 
            predicted_color_256 = predicted_color_256 * mask[..., None, :, :] 
            predicted_color_128 = predicted_color_128 * mask[..., None, :, :] 

        if predicted_color_512.shape[1] != target_color.shape[1]:
            print(f"Warning: predicted_color.shape[1] ({predicted_color.shape[1]}) != target_color.shape[1] ({target_color.shape[1]}), reshaping predicted_color")
            predicted_color = predicted_color[:, :target_color.shape[1], ...]
            
        # print(f"target_color.shape (b v c h w): {target_color.shape}, predicted_color.shape: {predicted_color.shape}, mask.shape: {mask.shape}")
        # flattened_color_512 = einops.rearrange(predicted_color_512, 'b v c h w -> (b v) c h w')
        # flattened_color_256 = einops.rearrange(predicted_color_256, 'b v c h w -> (b v) c h w')
        # flattened_color_128 = einops.rearrange(predicted_color_128, 'b v c h w -> (b v) c h w')
        # flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        # flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        # MSE loss
        rgb_l2_loss_512 = ((predicted_color_512 - target_color) ** 2)
        rgb_l2_loss_256 = ((predicted_color_256 - target_color) ** 2)
        rgb_l2_loss_128 = ((predicted_color_128 - target_color) ** 2)
        rgb_l2_loss_512 = rgb_l2_loss_512.mean(dim=2) * (1.0 + self.config.mse_penalty_512) # (b, v, c, h, w) -> (b, v, h, w)
        rgb_l2_loss_256 = rgb_l2_loss_256.mean(dim=2) * (1.0 + self.config.mse_penalty_256) # (b, v, c, h, w) -> (b, v, h, w)
        rgb_l2_loss_128 = rgb_l2_loss_128.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
        # print(f"rgb_l2_loss_512 shape: {rgb_l2_loss_512.shape}, rgb_l2_loss_256 shape: {rgb_l2_loss_256.shape}, rgb_l2_loss_128 shape: {rgb_l2_loss_128.shape}")
        rgb_l2_losses = torch.stack([rgb_l2_loss_512, rgb_l2_loss_256, rgb_l2_loss_128], dim=2) # (b, v, 3, h, w)
        # coarseness_gt = torch.zeros_like(rgb_l2_losses)
        # coarseness_gt.scatter_(2, torch.argmin(rgb_l2_losses, dim=2, keepdim=True), 1) # (b, v, 3, h, w)
        coarseness_gt = torch.argmin(rgb_l2_losses, dim=2) # (b, v, h, w)

        coarseness_gt = self.get_coarseness_gt(predicted_color_512, predicted_color_256, predicted_color_128, target_color)
        # coarseness_gt = coarseness_gt.squeeze(1) # (b, h, w)
        
        print(f"rgb_l2_losses: {rgb_l2_losses.shape}")

        # coarsenesses_image: # (b, v, c, h, w)
        one_hot_pred_coarseness = torch.zeros_like(coarsenesses_image)
        one_hot_pred_coarseness.scatter_(2, torch.argmax(coarsenesses_image, dim=2, keepdim=True), 1) # (b, v, 3, h, w)
        num_gaussians = 16*one_hot_pred_coarseness[:,:,0,...].sum() + 4*one_hot_pred_coarseness[:,:,1,...].sum() + 1*one_hot_pred_coarseness[:,:,2,...].sum()

        # print(f"num_gaussians: {num_gaussians.item()}")
        # coarseness_gt = torch.zeros_like(rgb_l2_losses)
        # coarseness_gt[torch.argmin(rgb_l2_losses, dim=2)[:, None, ...]] = 1.0
        # coarseness_gt = torch.argmin(rgb_l2_losses, dim=2)
        coarseness_gt = einops.rearrange(coarseness_gt, 'b v h w -> (b v) h w')
        # print(f"coarseness_gt: {coarseness_gt.shape}, ") # (1, h, w)
        coarsenesses_image = einops.rearrange(coarsenesses_image, 'b v c h w -> (b v) c h w')
        # print(f"coarsenesses_image: {coarsenesses_image.shape}, ") # (1, 3, h, w)
        # print(f"coarseness_image: \n \n {coarsenesses_image}")
        # print(f"coarseness_gt: \n \n {coarseness_gt}")
        # coarsenesses_image = torch.softmax(coarsenesses_image, dim=2) # (b, v, c, h, w)
        # print("means of coarseness images: ", coarsenesses_image.mean())

        if get_gt_coarseness:
            one_hot_labels = torch.nn.functional.one_hot(coarseness_gt, num_classes=3)
            return one_hot_labels
        weights = []
        loss = torch.tensor(0.0, device=coarseness_gt.device, dtype=torch.float32, requires_grad=True)
        for v in range(coarseness_gt.shape[0]):
            num_512, num_256, num_128 = torch.count_nonzero(coarseness_gt[v, ...] == 0), torch.count_nonzero(coarseness_gt[v, ...] == 1), torch.count_nonzero(coarseness_gt[v, ...] == 2)
            # print(f"num_512: {num_512.item()}, num_256: {num_256.item()}, num_128: {num_128.item()}")
            weight_512, weight_256, weight_128 = 1.0/num_512.item() if num_512.item() > 0 else 0.0, 1.0/num_256.item() if num_256.item() > 0 else 0.0, 1.0/num_128.item() if num_128.item() > 0 else 0.0
            weight_sum = weight_512 + weight_256 + weight_128
            weights = torch.tensor([weight_512/weight_sum, weight_256/weight_sum, weight_128/weight_sum], device=coarseness_gt.device, dtype=torch.float32) # (3,)
            # print(f"weights: {weights}")
            cross_entropy_loss = torch.nn.functional.cross_entropy(coarsenesses_image[v, ...].unsqueeze(0), coarseness_gt[v, ...].unsqueeze(0), weight=weights, reduction='none') # (h, w)
            
            if average_over_mask:
                # print("averaging")
                cross_entropy_loss = (cross_entropy_loss * mask[:, v, ...].unsqueeze(1)).sum() / mask.sum()
            else:
                cross_entropy_loss = cross_entropy_loss.mean()
            loss = loss + cross_entropy_loss
            
            # weights.append([weight_512/weight_sum, weight_256/weight_sum, weight_128/weight_sum])
        # weights = torch.tensor(weights, device=coarseness_gt.device, dtype=torch.float32) # (bv, 3)
        
        # The one-hot encoding creates a new dimension at the *end*, so we permute it
        # to match the standard PyTorch format [N, C, H, W].
        # one_hot_labels_permuted = one_hot_labels.permute(0, 1, 4, 2, 3).float()
        # one_hot_labels_permuted = einops.rearrange(one_hot_labels, 'b v h w c -> (b v) c h w')
        # print(f"\nOne-hot encoded labels shape: {one_hot_labels_permuted.shape}")
        # classification_loss = self.crossentropy_criterion(coarsenesses_image, coarseness_gt) # (b, v, h, w)
        # print(f"classification_loss: {classification_loss}, {classification_loss.shape}")
        # print(f"mask.shape: {mask.shape}, mask.sum(): {mask.sum()}")
        # if average_over_mask:
        #     print("averaging")
        #     classification_loss = (classification_loss * mask).sum() / mask.sum()
        # else:
        #     classification_loss = classification_loss.mean()

        del predicted_color_512, predicted_color_256, predicted_color_128
        print(f"classification_loss: {loss.item()}, {loss.shape}")
        return loss, num_gaussians

    def create_coarseness_gaussians(self, pred1_512, pred2_512, coarseness1, coarseness2):
        pred1_coarseness, pred2_coarseness = {}, {}
        for key in ["opacities", "rotations", "scales", "covariances", "depth_mask"]:
            pred1_coarseness[key] = pred1_512[key].detach()
            pred2_coarseness[key] = pred2_512[key].detach()
        pred1_coarseness['means'] = pred1_512['means'].detach()
        pred2_coarseness['means_in_other_view'] = pred2_512['means_in_other_view'].detach()
        # print(f"shape sh = {pred1_512['sh'].shape}, {pred2_512['sh'].shape}")
        # print(f"shape coarseness1 = {coarseness1.shape}, coarseness2 = {coarseness2.shape}")
        pred1_coarseness['sh'] = einops.rearrange(coarseness1, "b c h w -> b h w c 1" ).requires_grad_(True)
        pred2_coarseness['sh'] = einops.rearrange(coarseness2, "b c h w -> b h w c 1" ).requires_grad_(True)
        return pred1_coarseness, pred2_coarseness

    def get_coarseness_gt(self, predicted_color_512, predicted_color_256, predicted_color_128, target_color):
        """
        Get the ground truth coarseness based on the predicted colors.
        Returns a tensor of shape (b, v, h, w)  with values 0, 1, or 2 indicating the coarseness level.
        """
        flattened_color_512 = einops.rearrange(predicted_color_512, 'b v c h w -> (b v) c h w')
        flattened_color_256 = einops.rearrange(predicted_color_256, 'b v c h w -> (b v) c h w')
        flattened_color_128 = einops.rearrange(predicted_color_128, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        ssim_512 = compute_ssim.compute_ssim(flattened_target_color, flattened_color_512, full=True).mean(dim=1)
        ssim_256 = compute_ssim.compute_ssim(flattened_target_color, flattened_color_256, full=True).mean(dim=1)
        ssim_128 = compute_ssim.compute_ssim(flattened_target_color, flattened_color_128, full=True).mean(dim=1)
        print(f"ssim_512.shape: {ssim_512.shape}, ssim_256.shape: {ssim_256.shape}, ssim_128.shape: {ssim_128.shape}")
        ssim_img = torch.stack([torch.tensor(ssim_512).float(), torch.tensor(ssim_256).float(), torch.tensor(ssim_128).float()], dim=-1)
        print(f"ssim_img.shape: {ssim_img.shape}")
        ssim_img[..., 0] = ssim_img[..., 0] * (1.0 - self.config.mse_penalty_512)  # Apply penalty for 512
        ssim_img[..., 1] = ssim_img[..., 1] * (1.0 - self.config.mse_penalty_256)  # Apply penalty for 256
        coarseness_gt = torch.argmax(ssim_img, dim=-1)  # ((b * v), h, w)
        coarseness_gt = einops.rearrange(coarseness_gt, '(b v) h w -> b v h w', b=predicted_color_512.shape[0], v=predicted_color_512.shape[1])
        return coarseness_gt


    def calculate_loss_coarseness_img_freq(self, view1, view2, pred1_512, pred2_512, coarseness1, coarseness2):
        valid1 = (pred1_512["conf"] > self.config.loss.mast3r_confidence_threshold)
        valid2 = (pred2_512["conf"] > self.config.loss.mast3r_confidence_threshold)
        coarseness_gt_1, coarseness_onehot_1 = lod_utils.get_3_stage_mask(view1['original_img'][0, ...], pred1_512["pts3d"][0, ..., -1], valid1[0, ...], self.device, 512, 512, th_depth=config.th_depth, th_rgb=config.th_rgb)
        coarseness_gt_2, coarseness_onehot_2 = lod_utils.get_3_stage_mask(view2['original_img'][0, ...], pred1_512["pts3d"][0, ..., -1], valid2[0, ...], self.device, 512, 512, th_depth=config.th_depth, th_rgb=config.th_rgb)
        # coarseness_gt_1.shape:  (1, 512, 512)
        # print(f"coarseness_gt_1.shape = {coarseness_gt_1.shape}, coarseness1.shape = {coarseness1.shape}")
        weights_1 = (512.0*512.0)/coarseness_onehot_1.sum(dim=(1,2))
        weights_2 = (512.0*512.0)/coarseness_onehot_2.sum(dim=(1,2))
        weights_normalized_1 = (weights_1) / torch.linalg.norm(weights_1)
        weights_normalized_2 = (weights_2) / torch.linalg.norm(weights_2)
        # print(f"weights: {weights}")
        crossentropy_criterion_1 = torch.nn.CrossEntropyLoss(reduction="none", weight=weights_normalized_1) # reduction='none'
        crossentropy_criterion_2 = torch.nn.CrossEntropyLoss(reduction="none", weight=weights_normalized_2) # reduction='none'
        classification_loss1 = crossentropy_criterion_1(coarseness1, coarseness_gt_1) # (b, v, h, w)
        classification_loss2 = crossentropy_criterion_2(coarseness2, coarseness_gt_2)
        # print(f"classification_loss: {classification_loss.shape}")
        loss = classification_loss1.mean() + classification_loss2.mean()
        # loss = classification_loss.mean()
        print(f"loss: {loss}")
        return loss


    def calculate_loss(self, batch, view1, view2, pred1, pred2, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):
        # print(f"len batch['target'] = {len(batch['target'])}")
        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color
        # print(f"predicted_color.shape: {predicted_color.shape}, target_color.shape: {target_color.shape}")
        if self.config.use_lod:
            num_gaussians = pred1['sh'].shape[1] + pred2['sh'].shape[1]
        else:
            num_gaussians = pred1['sh'].shape[1]*pred1['sh'].shape[2] + pred2['sh'].shape[1]*pred2['sh'].shape[2]
        if apply_mask:
            if mask.sum() < 1:
                print(f"Skipping batch due to no valid pixels in the mask! batch['scene_id'] = {batch['scene']}")
                zero_loss = torch.tensor(0.0, device=target_color.device, requires_grad=True)
                if calculate_ssim:
                    return zero_loss, zero_loss, zero_loss, zero_loss
                return zero_loss, zero_loss, zero_loss, 0
            target_color = target_color * mask[..., None, :, :]
            # print(f"predicted_color.shape: {predicted_color.shape}, target_color.shape: {target_color.shape}, mask.shape: {mask.shape}")
            predicted_color = predicted_color * mask[..., None, :, :]
        if predicted_color.shape[1] != target_color.shape[1]:
            print(f"Warning: predicted_color.shape[1] ({predicted_color.shape[1]}) != target_color.shape[1] ({target_color.shape[1]}), reshaping predicted_color")
            predicted_color = predicted_color[:, :target_color.shape[1], ...]
        # print(f"target_color.shape (b v c h w): {target_color.shape}, predicted_color.shape: {predicted_color.shape}, mask.shape: {mask.shape}")
        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')
        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        # print(f"flattened_color.shape: {flattened_color.shape}, flattened_target_color.shape: {flattened_target_color.shape}")
        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        # Calculate the total loss
        loss = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss

        # MAST3R Loss
        if self.config.loss.mast3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        # Masked SSIM
        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val, num_gaussians

        return loss, mse_loss, lpips_loss, num_gaussians

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None, num_gaussians=None, train_coarseness_prediction=False):
        if train_coarseness_prediction:
            values = {
                f'{prefix}/loss': loss,
                f'{prefix}/num_gaussians': num_gaussians
            }
        elif self.config.train_3stage_gaussian_heads:
            mse_512, mse_256, mse_128 = mse
            lpips_512, lpips_256, lpips_128 = lpips
            loss_512, loss_256, loss_128 = loss
            loss_avg = (loss_512 + loss_256 + loss_128) / 3.0
            values = {
                f'{prefix}/loss_avg': loss_avg,
                f'{prefix}/mse_512': mse_512,
                f'{prefix}/mse_256': mse_256,
                f'{prefix}/mse_128': mse_128,
                f'{prefix}/lpips_512': lpips_512,
                f'{prefix}/lpips_256': lpips_256,
                f'{prefix}/lpips_128': lpips_128,
                f'{prefix}/psnr_512': -10.0 * mse_512.log10(),
                f'{prefix}/psnr_256': -10.0 * mse_256.log10(),
                f'{prefix}/psnr_128': -10.0 * mse_128.log10(),
            }

        else:
            values = {
                f'{prefix}/loss': loss,
                f'{prefix}/mse': mse,
                f'{prefix}/psnr': -10.0 * mse.log10(),
                f'{prefix}/lpips': lpips,
                f'{prefix}/num_gaussians': num_gaussians
            }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def load_gaussian_head_params(self, path):
        gaussian_head_params = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        
        # Filter out unnecessary keys and update the model's state_dict
        gaussian_head_params = {k: v for k, v in gaussian_head_params.items() if k in model_dict}
        model_dict.update(gaussian_head_params)

        # Load the updated state_dict into the model
        self.load_state_dict(model_dict)

def run_experiment(config):

    # Set the seed
    L.seed_everything(config.seed, workers=True)

    print(f"Training {config.resolution} Head")

    # Set up loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.use_lod:
        name_suffix = "_lod"
    else:
        name_suffix = f"_{config.resolution}"
    name = config.name + name_suffix
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=name
        )
        loggers.append(csv_logger)
    wandb_project = 'splatt3r' if not config.train_coarseness_prediction else 'splatt3r_coarseness_pred'
    if config.loggers.use_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project=wandb_project,
            name=name,
            save_dir=config.save_dir,
            config=omegaconf.OmegaConf.to_container(config),
        )
        if wandb.run is not None:
            wandb.run.log_code(".")
        loggers.append(wandb_logger)

    # Set up profiler
    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler = None

    # Model
    print('Loading Model')
    if config.use_pretrained:
        # ckpt = torch.load(config.pretrained_mast3r_path)
        # print(f"ckpt keys: {ckpt.keys()}")
        # _ = model.encoder.load_state_dict(ckpt['state_dict'], strict=False)
        # del ckpt
        model = MAST3RGaussians.load_from_checkpoint(checkpoint_path=config.pretrained_mast3r_path, device='cuda:0', config=config, strict=False)
    else:
        model = MAST3RGaussians(config)


    # Training Datasets
    print(f'Building Datasets')
    train_dataset = scannetpp.get_scannet_dataset(
        config.data.root,
        'train',
        config.data.resolution,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch,
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    val_dataset = scannetpp.get_scannet_test_dataset(
        config.data.root,
        alpha=0.5,
        beta=0.5,
        resolution=config.data.resolution,
        use_every_n_sample=100,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Training
    print('Training')
    coarse = "coarse" if config.resolution < 500 else ""
    val_every_n_epochs = 4
    print("save_dir:", config.save_dir)
    date_time = config.save_dir[-18:-1]
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_file_path, # <--- specify this on the trainer itself for version control
        filename= date_time + "_{epoch:02d}_{step:05d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,  # <--- this is important!
    )

    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            export.SaveBatchData(save_dir=config.save_dir, train_save_interval=300, coarse=config.resolution < 500, lod=config.use_lod, train_coarse_prediction=config.train_coarseness_prediction, coarseness_predictions=config.coarseness_predictions),
            checkpoint_callback
        ],
        check_val_every_n_epoch=1,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=10, # 10
        logger=loggers,
        max_epochs=config.opt.epochs,
        profiler=profiler,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
        # overfit_batches=0,
        # enable_checkpointing=False
    )
    # trainer.validate(model, dataloaders=data_loader_val)
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

    # Testing
    original_save_dir = config.save_dir
    results = {}

    return
    for alpha, beta in ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5), (0.3, 0.3)):

        test_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=alpha,
            beta=beta,
            resolution=config.data.resolution,
            use_every_n_sample=100
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )

        masking_configs = ((True, False), (True, True))
        for apply_mask, average_over_mask in masking_configs:

            new_save_dir = os.path.join(
                original_save_dir,
                f'alpha_{alpha}_beta_{beta}_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}'
            )
            os.makedirs(new_save_dir, exist_ok=True)
            model.config.save_dir = new_save_dir

            L.seed_everything(config.seed, workers=True)

            # Training
            trainer = L.Trainer(
                accelerator="gpu",
                benchmark=True,
                callbacks=[export.SaveBatchData(save_dir=config.save_dir, coarse=config.resolution < 500, lod=config.use_lod, train_coarse_prediction=config.train_coarseness_prediction),],
                default_root_dir=config.save_dir,
                devices=config.devices,
                log_every_n_steps=10,
                strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
            )

            model.lpips_criterion = lpips.LPIPS('vgg', spatial=average_over_mask)
            model.config.loss.apply_mask = apply_mask
            model.config.loss.average_over_mask = average_over_mask
            res = trainer.test(model, dataloaders=data_loader_test)
            results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res
            # Save the results
            save_path = os.path.join(original_save_dir, 'results.json')
            with open(save_path, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run training
    run_experiment(config)
