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
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

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
            self.crossentropy_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.save_hyperparameters()

    def forward(self, view1, view2):

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1, pred1_256, pred1_128, coarseness1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2, pred2_256, pred2_128, coarseness2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        pred1_256['covariances'] = geometry.build_covariance(pred1_256['scales'], pred1_256['rotations'])
        pred2_256['covariances'] = geometry.build_covariance(pred2_256['scales'], pred2_256['rotations'])

        pred1_128['covariances'] = geometry.build_covariance(pred1_128['scales'], pred1_128['rotations'])
        pred2_128['covariances'] = geometry.build_covariance(pred2_128['scales'], pred2_128['rotations'])

        # print(f"pred1_256['covariances']: {pred1_256['covariances'].shape}")
        # print(f"pred1_256['opacities']: {pred1_256['opacities'].shape}")
        # pred1_256['covariances'] = pred1['covariances'][:,::2,::2,:]
        # pred2_256['covariances'] = pred2['covariances'][:,::2,::2,:]
        # pred1_256['opacities'] = pred1['opacities'][:,::2,::2,:]
        # pred1_256['opacities'] = pred1['opacities'][:,::2,::2,:]            
        
        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            # print(f"original image max: {view1['original_img'].max()}, min: {view1['original_img'].min()}")
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

            sh1_256 = (new_sh1[:,::2,::2,:] + new_sh1[:,1::2,::2,:] + new_sh1[:,::2,1::2,:] + new_sh1[:,1::2,1::2,:])/ 4.0
            sh2_256 = (new_sh2[:,::2,::2,:] + new_sh2[:,1::2,::2,:] + new_sh2[:,::2,1::2,:] + new_sh2[:,1::2,1::2,:])/ 4.0
            pred1_256['sh'] = sh1_256
            pred2_256['sh'] = sh2_256

            sh1_128 = (sh1_256[:,::2,::2,:] + sh1_256[:,1::2,::2,:] + sh1_256[:,::2,1::2,:] + sh1_256[:,1::2,1::2,:])/ 4.0
            sh2_128 = (sh2_256[:,::2,::2,:] + sh2_256[:,1::2,::2,:] + sh2_256[:,::2,1::2,:] + sh2_256[:,1::2,1::2,:])/ 4.0

            pred1_128['sh'] = pred1_128['sh'] + sh1_128
            pred2_128['sh'] = pred2_128['sh'] + sh2_128

        if self.config["use_calib"]:
            # assert False, "constrain to ray with T_21"
            intrinsics1 = view1['camera_intrinsics']
            intrinsics2 = view2['camera_intrinsics']
            T_wc1 = view1['camera_pose'] # c2w
            T_wc2 = view2['camera_pose']
            print(f"extinsics1: {T_wc1}")
            T_c1_c2 = torch.linalg.inv(T_wc1) @ T_wc2 # T_wc1^-1 @ T_wc2 = T_c1c2
            depth1 = view1['depthmap']
            depth2 = view2['depthmap']
            assert intrinsics1.shape[0] == 1, "assume single batch"
            assert intrinsics2.shape[0] == 1, "assume single batch"
            intrinsics1, intrinsics2 = intrinsics1[0,...], intrinsics2[0,...]
            h, w = view1['original_img'].shape[-2:]
            print(f"h, w = {h}, {w}")
            print(f"inrinsics__: {intrinsics1[0,0]}, {intrinsics1[1,1]}, {intrinsics1[0,2]}, {intrinsics1[1,2]}")

            print(f"pred1[means].shape = {pred1['means'].shape}, pred2[means].shape = {pred2['means'].shape}")
            print(f"depth1.shape = {depth1.shape}, depth2.shape = {depth2.shape}")
            pred1['means'] = depth_map_to_points((h, w), depth1, intrinsics1)
            points2_c2 = depth_map_to_points((h, w), depth2, intrinsics2) # (b, h, w, 3)
            points2_homogeneous_c2 = torch.cat([points2_c2, torch.ones_like(points2_c2[..., :1])], dim=-1)

            points2_c1 = torch.einsum('bij, bhwj -> bhwi', T_c1_c2, points2_homogeneous_c2)
            pred2['means'] = points2_c1[..., :3] # (b, h, w, 3)

            pred1['pts3d'] = pred1['means']
            pred2['pts3d'] = pred2['means']

        # print(f"max sh1: {pred1['sh'].max()}, min sh1: {pred1['sh'].min()}")
        means1 = pred1['means']
        means2 = pred2['means']
        # print(f"means1 shape = {means1.shape}, means2 shape = {means2.shape}")
        pred1_256['means'] = (means1[:,::2,::2,:] + means1[:,1::2,::2,:] + means1[:,::2,1::2,:] + means1[:,1::2,1::2,:]) / 4.0
        pred2_256['means_in_other_view'] = (means2[:,::2,::2,:] + means2[:,1::2,::2,:] + means2[:,::2,1::2,:] + means2[:,1::2,1::2,:]) / 4.0
        
        means1_256 = pred1_256['means']
        means2_256 = pred2_256['means_in_other_view']

        pred1_128['means'] = (means1_256[:,::2,::2,:] + means1_256[:,1::2,::2,:] + means1_256[:,::2,1::2,:] + means1_256[:,1::2,1::2,:]) / 4.0
        pred2_128['means_in_other_view'] = (means2_256[:,::2,::2,:] + means2_256[:,1::2,::2,:] + means2_256[:,::2,1::2,:] + means2_256[:,1::2,1::2,:]) / 4.0

        if self.config.use_lod:
            # print(f"view1['original_img'].shape = {view1['original_img'].shape}")
            # print(f"pred1['pts3d'].shape = {pred1['pts3d'].shape}")
            H,W = view1['original_img'].shape[2:]
            # print(f"H,W = {H}, {W}")
            confidence_threshold = 1.5
            # print(f"pred1.keys() = {pred1.keys()}")
            valid = (pred1["conf"] > confidence_threshold)
            valid2 = (pred2["conf"] > confidence_threshold)
            device = "cuda:0"

            th_rgb = self.config.th_rgb
            th_depth = self.config.th_depth
            mask1_256 = torch.zeros((pred1["pts3d"].shape[0], H//2, W//2), dtype=torch.bool, device=device)
            mask2_256 = torch.zeros((pred2["pts3d"].shape[0], H//2, W//2), dtype=torch.bool, device=device)
            mask1 = torch.zeros((pred1["pts3d"].shape[0], H, W), dtype=torch.bool, device=device)
            mask2 = torch.zeros((pred2["pts3d"].shape[0], H, W), dtype=torch.bool, device=device)
            for b in range(pred1["pts3d"].shape[0]):
                # print(f"batch {b}, pred1['pts3d'].shape = {pred1['pts3d'].shape}, pred1['means'].shape = {pred1['means'].shape} valid.shape = {valid.shape}")
                mask1_256[b, ...], mask1[b, ...] =  lod_utils.get_mask(view1['original_img'][b, ...], pred1["pts3d"][b, ..., -1], valid[b, ...], device, H, W, th_depth=th_depth, th_rgb=th_rgb)
                
                # print(f"mask1_256.shape = {mask1_256.shape}, mask1.shape = {mask1.shape}")
                # print(f"mask1_256[b, ...].shape = {mask1_256[b, ...].shape}, mask1[b, ...].shape = {mask1[b, ...].shape}")

                mask2_256[b, ...] , mask2[b, ...] =  lod_utils.get_mask(view2['original_img'][b, ...], pred2["pts3d"][b, ..., -1], valid2[b, ...], device, H, W, th_depth=th_depth, th_rgb=th_rgb)
                
                # mask1_256[b, ...], mask1[b, ...] = lod_utils.get_mask(view1['original_img'][b, ...], pred1["pts3d"][b, ..., -1], valid, device, H, W, th_depth=th_depth, th_rgb=th_rgb)
                # mask2_256[b, ...], mask2[b, ...] = lod_utils.get_mask(view2['original_img'][b, ...], pred2["pts3d"][...,-1][b, ...], valid, device, H, W, th_depth=th_depth, th_rgb=th_rgb)
            pred1_combined = lod_utils.apply_mask_to_gaussians(pred1, pred1_256, mask1, mask1_256)
            pred2_combined = lod_utils.apply_mask_to_gaussians(pred2, pred2_256, mask2, mask2_256)
            pred2_combined['means_in_other_view'] = pred2_combined.pop('means')
            pred1_combined['mask_highres'] = mask1
            pred2_combined['mask_highres'] = mask2

        if self.config.coarseness_predictions:
            
            predicted_classes = ((pred1, pred1_256, pred1_128, coarseness1, None), (pred2, pred2_256, pred2_128, coarseness2, None))
            for pred, pred_256, pred_128, coarseness, pred_combined in predicted_classes:
                classes = torch.argmax(coarseness, dim=-1)
                mask_256 = torch.logical_and(classes[:,::2,::2], classes[:,1::2,::2], classes[:,::2,1::2], classes[:,1::2,1::2])
                mask_128 = torch.logical_and(mask1_256[:,::2,::2], mask1_256[:,1::2,::2], mask1_256[:,::2,1::2], mask1_256[:,1::2,1::2])
                mask_128_upsampled_256 =  torch.nn.functional.interpolate(mask_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                mask_256 = mask_256 and not mask_128_upsampled_256
                mask_256_upsampled_512 = torch.nn.functional.interpolate(mask_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
                mask_512 = not mask_256_upsampled_512
                for key in pred.keys():
                    if key not in ["means", "opacities", "sh", "rotations", "scales", "covariances"]:
                        continue
                    tensor_list = []
                    for b in range(pred["means"].shape[0]):
                        # print(f"fusing key, {key}, pred[key].shape: {pred[key].shape}, pred_lowres[key].shape: {pred_lowres[key].shape}")
                        tensor_list.append(torch.cat([pred[key][b,mask_512[b,...],...], 
                                                      pred_256[key][b,mask_256[b,...],...], 
                                                      pred_128[b,mask_512[b,...]]]
                                                      , dim=0))
                    pred_combined[key] = torch.stack(tensor_list, dim=0)
                

        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')
        # for key in pred2_combined.keys():
            # print(f"key. {key}, pred2_combined[key].shape = {pred2_combined[key].shape}")
        if self.config.train_coarseness_prediction:
            return pred1, pred2, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2
        if self.config.use_lod or self.config.coarseness_predictions:
            # return pred1, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
            return pred1_combined, pred2_combined
        else:
            return pred1, pred2, pred1, pred2, pred1_128, pred2_128
        
    def training_step(self, batch, batch_idx):
        num_targets = len(batch['target'])
        # print(f"Training step {batch_idx}, len batch context {len(batch['context'])}, num targets: {num_targets}")
        print(f"Training scene {batch['scene']}, step {batch_idx}")
        # print(f"Training step {batch_idx}, batch size: {len(batch['context'])}")
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        if self.config.train_coarseness_prediction:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            with torch.no_grad():
                color_512, _ = self.decoder(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
                color_256, _ = self.decoder(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                color_128, _ = self.decoder(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
            pred1_coarseness, pred2_coarseness = {}, {}
            for key in ["opacities", "rotations", "scales", "covariances"]:
                pred1_coarseness[key] = pred1_512[key].clone().requires_grad_(False)
                pred2_coarseness[key] = pred2_512[key].clone().requires_grad_(False)
            pred1_coarseness['means'] = pred1_512['means'].clone().requires_grad_(False)
            pred2_coarseness['means_in_other_view'] = pred2_512['means_in_other_view'].clone().requires_grad_(False)
            print(f"shape sh = {pred1_512['sh'].shape}, {pred2_512['sh'].shape}")
            print(f"shape coarseness1 = {coarseness1.shape}, coarseness2 = {coarseness2.shape}")
            pred1_coarseness['sh'] = einops.rearrange(coarseness1, "b c h w -> b h w c 1" )
            pred2_coarseness['sh'] = einops.rearrange(coarseness2, "b c h w -> b h w c 1" )

            coarseness_image, _ = self.decoder(batch, pred1_coarseness, pred2_coarseness, (h, w)) # gets (b, v, c, h, w)
            # coarseness_image, _ = self.decoder(batch, pred1_512, pred2_512, (h, w))

            print(f"coarseness_image.shape = {coarseness_image.shape}, pred1_coarseness['sh'].shape = {pred1_coarseness['sh'].shape}, pred2_coarseness['sh'].shape = {pred2_coarseness['sh'].shape}")
            mask = loss_mask.calculate_loss_mask(batch)
            print(f"coarseness_image: {coarseness_image}")
            
            loss, num_gaussians = self.calculate_loss_3stage(batch, view1, view2,
                                       (pred1_512, pred1_256, pred1_128),
                                       (pred2_512, pred2_256, pred2_128),
                                       (color_512, color_256, color_128),
                                       coarseness_image,
                                        mask)
            
            # loss, mse, lpips, num_gaussians = self.calculate_loss(
            #     batch, view1, view2, pred1_256, pred2_256, color, mask,
            #     apply_mask=self.config.loss.apply_mask,
            #     average_over_mask=self.config.loss.average_over_mask,
            #     calculate_ssim=False
            # )
            self.log_metrics('train', loss, mse=0, lpips=0, num_gaussians=num_gaussians, train_coarseness_prediction=True)
            return loss

        elif self.config.use_lod or self.config.coarseness_predictions:
            # return pred1, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
            pred1_combined, pred2_combined = self.forward(view1, view2)
        else:
            pred1, pred2, pred1, pred2, pred1_128, pred2_128 = self.forward(view1, view2)
        
        if self.config.resolution < 500:
            color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w))
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
        elif self.config.use_lod:
            color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w), fused_gaussians=True)
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
        else:
            color, _ = self.decoder(batch, pred1, pred2, (h, w))
        
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1, pred2, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )

        # Log losses
        self.log_metrics('train', loss, mse, lpips, num_gaussians=num_gaussians)
        return loss

    def validation_step(self, batch, batch_idx):
        num_targets = len(batch['target'])
        # print(f"Validation scene {batch['scene']}, len batch context {len(batch['context'])}, {batch['context'][0].keys()} num targets: {num_targets}")
        print(f"validation step {batch['scene']}")
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        
        # Predict using the encoder/decoder and calculate the loss
        if self.config.train_coarseness_prediction:
            # Predict using the encoder/decoder and calculate the loss
            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = self.forward(view1, view2)
            with torch.no_grad():
                color_512, _ = self.decoder(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
                color_256, _ = self.decoder(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                color_128, _ = self.decoder(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
                print(f"color_512.shape = {color_512.shape}, color_256.shape = {color_256.shape}, color_128.shape = {color_128.shape}")
            pred1_coarseness, pred2_coarseness = {}, {}
            for key in ["opacities", "rotations", "scales", "covariances"]:
                pred1_coarseness[key] = pred1_512[key].clone().requires_grad_(False)
                pred2_coarseness[key] = pred2_512[key].clone().requires_grad_(False)
            pred1_coarseness['means'] = pred1_512['means'].clone().requires_grad_(False)
            pred2_coarseness['means_in_other_view'] = pred2_512['means_in_other_view'].clone().requires_grad_(False)
            print(f"pred1_coarseness['means'].shape = {pred1_coarseness['means'].shape}, pred2_coarseness['means_in_other_view'].shape = {pred2_coarseness['means_in_other_view'].shape}")
            pred1_coarseness['sh'] = einops.rearrange(coarseness1, "b c h w -> b h w c 1" ).requires_grad_(True)
            pred2_coarseness['sh'] = einops.rearrange(coarseness2, "b c h w -> b h w c 1" ).requires_grad_(True)

            coarseness_image, _ = self.decoder(batch, pred1_coarseness, pred2_coarseness, (h, w)) # gets (b, v, c, h, w)
            print(f"coarseness_image: {coarseness_image}")
            print(f"color_512: {color_512}")
            print(f"coarseness_image.shape = {coarseness_image.shape}, pred1_coarseness['sh'].shape = {pred1_coarseness['sh'].shape}, pred2_coarseness['sh'].shape = {pred2_coarseness['sh'].shape}")
            mask = loss_mask.calculate_loss_mask(batch)
            loss, num_gaussians = self.calculate_loss_3stage(batch, view1, view2,
                                       (pred1_512, pred1_256, pred1_128),
                                       (pred2_512, pred2_256, pred2_128),
                                       (color_512, color_256, color_128),
                                       coarseness_image,
                                        mask)
            
            # loss, mse, lpips, num_gaussians = self.calculate_loss(
            #     batch, view1, view2, pred1_256, pred2_256, color, mask,
            #     apply_mask=self.config.loss.apply_mask,
            #     average_over_mask=self.config.loss.average_over_mask,
            #     calculate_ssim=False
            # )
            self.log_metrics('val', loss, mse=0, lpips=0, num_gaussians=num_gaussians, train_coarseness_prediction=True)
            return loss

        elif self.config.use_lod or self.config.coarseness_predictions:
            # return pred1, pred2, pred1_256, pred2_256, pred1_combined, pred2_combined
            pred1_combined, pred2_combined = self.forward(view1, view2)
        else:
            pred1, pred2, pred1, pred2, pred1_128, pred2_128 = self.forward(view1, view2)
        
        if self.config.resolution < 500:
            color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w))
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
        elif self.config.use_lod:
            color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w), fused_gaussians=True)
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )
        else:
            color, _ = self.decoder(batch, pred1, pred2, (h, w))
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1, pred2, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=False
            )

        # print(f"Loss: {loss.item()}, MSE: {mse.item()}, LPIPS: {lpips.item()} ------------------------------------")
        # Log losses
        self.log_metrics('val', loss, mse, lpips, num_gaussians=num_gaussians)
        return loss

    def test_step(self, batch, batch_idx):
        print(f"test step {batch_idx}, scene {batch['scene']}")
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        num_targets = len(batch['target'])

        # Predict using the encoder/decoder and calculate the loss
        with self.benchmarker.time("encoder"):
            pred1, pred2, pred1_256, pred2_256 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            if self.config.resolution < 500:
                color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w))
            elif self.config.use_lod:
                color, _ = self.decoder(batch, pred1_256, pred2_256, (h, w), fused_gaussians=True)
            else:
                color, _ = self.decoder(batch, pred1, pred2, (h, w))
        
        if self.config.resolution < 500 or self.config.use_lod:
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, ssim, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1_256, pred2_256, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=True
            )
        else:
            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse, lpips, ssim, num_gaussians = self.calculate_loss(
                batch, view1, view2, pred1, pred2, color, mask,
                apply_mask=self.config.loss.apply_mask,
                average_over_mask=self.config.loss.average_over_mask,
                calculate_ssim=True
            )

        # Log losses
        self.log_metrics('test', loss, mse, lpips, ssim=ssim, num_gaussians=num_gaussians)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss_3stage(self, batch, view1, view2, pred1, pred2, colors, coarsenesses_image, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):
        # print(f"len batch['target'] = {len(batch['target'])}")
        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        (predicted_color_512, predicted_color_256, predicted_color_128) = colors
        
        print(f"predicted_color_512.shape: {predicted_color_512.shape}, target_color.shape: {target_color.shape}")
        # if self.config.use_lod:
        #     num_gaussians = pred1['sh'].shape[1] + pred2['sh'].shape[1]
        # else:
        #     num_gaussians = pred1['sh'].shape[1]*pred1['sh'].shape[2] + pred2['sh'].shape[1]*pred2['sh'].shape[2]
        coarseness_image_empty = coarsenesses_image.mean(dim=2) # (b, v, h, w)
        mask_coarseness_render = torch.ones_like(coarseness_image_empty)
        mask_coarseness_render[coarseness_image_empty == 0] = 0 # (b, v, h, w)
        mask = mask * mask_coarseness_render
        if apply_mask:
            if mask.sum() < 1:
                print(f"Skipping batch due to no valid pixels in the mask! batch['scene_id'] = {batch['scene']}")
                zero_loss = torch.tensor(0.0, device=target_color.device, requires_grad=True)
                if calculate_ssim:
                    return zero_loss, zero_loss, zero_loss, zero_loss
                return zero_loss, zero_loss, zero_loss, 0
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
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        # MSE loss
        rgb_l2_loss_512 = ((predicted_color_512 - target_color) ** 2)
        rgb_l2_loss_256 = ((predicted_color_256 - target_color) ** 2)
        rgb_l2_loss_128 = ((predicted_color_128 - target_color) ** 2)
        rgb_l2_loss_512 = rgb_l2_loss_512.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
        rgb_l2_loss_256 = rgb_l2_loss_256.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
        rgb_l2_loss_128 = rgb_l2_loss_128.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
        print(f"rgb_l2_loss_512 shape: {rgb_l2_loss_512.shape}, rgb_l2_loss_256 shape: {rgb_l2_loss_256.shape}, rgb_l2_loss_128 shape: {rgb_l2_loss_128.shape}")
        rgb_l2_losses = torch.stack([rgb_l2_loss_512, rgb_l2_loss_256, rgb_l2_loss_128], dim=2) # (b, v, 3, h, w)
        coarseness_gt = torch.zeros_like(rgb_l2_losses)
        coarseness_gt.scatter_(2, torch.argmin(rgb_l2_losses, dim=2, keepdim=True), 1) # (b, v, 3, h, w)
        print(f"rgb_l2_losses: {rgb_l2_losses.shape}")

        # coarsenesses_image: # (b, v, c, h, w)
        one_hot_pred_coarseness = torch.zeros_like(coarsenesses_image)
        one_hot_pred_coarseness.scatter_(2, torch.argmax(coarsenesses_image, dim=2, keepdim=True), 1) # (b, v, 3, h, w)
        num_gaussians = 16*one_hot_pred_coarseness[:,:,0,...].sum() + 4*one_hot_pred_coarseness[:,:,1,...].sum() + 1*one_hot_pred_coarseness[:,:,2,...].sum()
        print(f"num_gaussians: {num_gaussians.item()}")
        # coarseness_gt = torch.zeros_like(rgb_l2_losses)
        # coarseness_gt[torch.argmin(rgb_l2_losses, dim=2)[:, None, ...]] = 1.0
        # coarseness_gt = torch.argmin(rgb_l2_losses, dim=2)
        # coarseness_gt = einops.rearrange(coarseness_gt, 'b v c h w -> b v h w c')
        print(f"coarseness_gt: {coarseness_gt.shape}, ")
        print(f"coarsenesses_image: {coarsenesses_image.shape}, ")
        print(f"")
        # print(f"coarseness_image: \n \n {coarsenesses_image}")
        # print(f"coarseness_gt: \n \n {coarseness_gt}")
        # coarsenesses_image = torch.softmax(coarsenesses_image, dim=2) # (b, v, c, h, w)
        print("means of coarseness images: ", coarsenesses_image.mean(), coarseness_gt.mean())
        classification_loss = self.crossentropy_criterion(coarsenesses_image, coarseness_gt) # (b, v, h, w)
        # print(f"classification_loss: {classification_loss}, {classification_loss.shape}")
        # print(f"mask.shape: {mask.shape}, mask.sum(): {mask.sum()}")
        if average_over_mask:
            print("averaging")
            classification_loss = (classification_loss * mask).sum() / mask.sum()
        else:
            classification_loss = classification_loss.mean()

        print(f"classification_loss: {classification_loss}")
        return classification_loss, num_gaussians
        # if average_over_mask:
        #     mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        # else:
        #     mse_loss = rgb_l2_loss.mean()

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
    val_every_n_epochs = 1
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_file_path, # <--- specify this on the trainer itself for version control
        filename="splatt3r_{coarse}_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,  # <--- this is important!
    )

    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            export.SaveBatchData(save_dir=config.save_dir, coarse=config.resolution < 500, lod=config.use_lod, train_coarse_prediction=config.train_coarseness_prediction),
            # checkpoint_callback
        ],
        check_val_every_n_epoch=1,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=10,
        logger=loggers,
        max_epochs=config.opt.epochs,
        profiler=profiler,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
    )
    # trainer.validate(model, dataloaders=data_loader_train)
    assert False, "crossentropyloss doesnt work on validation data, check for NaNs or sth"
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

    # Testing
    original_save_dir = config.save_dir
    results = {}
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
