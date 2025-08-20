import os

from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import einops
import numpy as np
import torch
import torchvision
import trimesh
import lightning as L
import utils.lod_utils as lod_utils

import utils.loss_mask as loss_mask
from src.mast3r_src.dust3r.dust3r.viz import OPENGL, pts3d_to_trimesh, cat_meshes


class SaveBatchData(L.Callback):
    '''A Lightning callback that occasionally saves batch inputs and outputs to disk.
    It is not critical to the training process, and can be disabled if unwanted.'''

    def __init__(self, save_dir, train_save_interval=100, val_save_interval=100, test_save_interval=100, coarse=True, grad_coarseness=False, train_coarse_prediction=False, coarseness_predictions=False):
        self.save_dir = save_dir
        self.train_save_interval = train_save_interval
        self.val_save_interval = val_save_interval
        self.test_save_interval = test_save_interval
        self.coarse = coarse
        self.grad_coarseness = grad_coarseness
        self.train_coarse_prediction = train_coarse_prediction
        self.coarseness_predictions = coarseness_predictions

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.train_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('train', trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.val_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('val', trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.test_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('test', trainer, pl_module, batch, batch_idx)

    def save_batch_data(self, prefix, trainer, pl_module, batch, batch_idx):

        print(f'Saving {prefix} data at epoch {trainer.current_epoch} and batch {batch_idx}')
        if self.train_coarse_prediction:
            # Run the batch through the model again
            _, _, h, w = batch["context"][0]["img"].shape
            view1, view2 = batch['context']
            target_views = batch['target']

            with torch.no_grad():
                pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = pl_module.forward(view1, view2)
                color_512, _ = pl_module.decoder(batch, pred1_512, pred2_512, (h, w)) # gets (b, v, c, h, w)
                color_256, _ = pl_module.decoder(batch, pred1_256, pred2_256, (h, w)) # gets (b, v, c, h, w)
                color_128, _ = pl_module.decoder(batch, pred1_128, pred2_128, (h, w)) # gets (b, v, c, h, w)
            colors = (color_512, color_256, color_128)

            # if apply_mask:
            #     if mask.sum() < 1:
            #         print(f"Skipping batch due to no valid pixels in the mask! batch['scene_id'] = {batch['scene']}")
            #         zero_loss = torch.tensor(0.0, device=target_color.device, requires_grad=True)
            #         if calculate_ssim:
            #             return zero_loss, zero_loss, zero_loss, zero_loss
            #         return zero_loss, zero_loss, zero_loss, 0
            #     target_color = target_color * mask[..., None, :, :]
            #     predicted_color_512 = predicted_color_512 * mask[..., None, :, :] 
            #     predicted_color_256 = predicted_color_256 * mask[..., None, :, :] 
            #     predicted_color_128 = predicted_color_128 * mask[..., None, :, :] 
            # if predicted_color_512.shape[1] != target_color.shape[1]:
            #     print(f"Warning: predicted_color.shape[1] ({predicted_color.shape[1]}) != target_color.shape[1] ({target_color.shape[1]}), reshaping predicted_color")
            #     predicted_color = predicted_color[:, :target_color.shape[1], ...]
            # print(f"target_color.shape (b v c h w): {target_color.shape}, predicted_color.shape: {predicted_color.shape}, mask.shape: {mask.shape}")
            # flattened_color_512 = einops.rearrange(predicted_color_512, 'b v c h w -> (b v) c h w')
            # flattened_color_256 = einops.rearrange(predicted_color_256, 'b v c h w -> (b v) c h w')
            # flattened_color_128 = einops.rearrange(predicted_color_128, 'b v c h w -> (b v) c h w')
            # flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
            # flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

            # # MSE loss
            # rgb_l2_loss_512 = ((color_512 - target_color) ** 2)
            # rgb_l2_loss_256 = ((color_256 - target_color) ** 2)
            # rgb_l2_loss_128 = ((color_128 - target_color) ** 2)
            # rgb_l2_loss_512 = rgb_l2_loss_512.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
            # rgb_l2_loss_256 = rgb_l2_loss_256.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
            # rgb_l2_loss_128 = rgb_l2_loss_128.mean(dim=2) # (b, v, c, h, w) -> (b, v, h, w)
            # print(f"rgb_l2_loss_512 shape: {rgb_l2_loss_512.shape}, rgb_l2_loss_256 shape: {rgb_l2_loss_256.shape}, rgb_l2_loss_128 shape: {rgb_l2_loss_128.shape}")
            # rgb_l2_losses = torch.stack([rgb_l2_loss_512, rgb_l2_loss_256, rgb_l2_loss_128], dim=2) # (b, v, 3, h, w)
            # coarseness_gt = torch.zeros_like(rgb_l2_losses)
            # coarseness_gt.scatter_(2, torch.argmin(rgb_l2_losses, dim=2, keepdim=True), 1) # (b, v, 3, h, w)

            pred1_512, pred2_512, pred1_256, pred2_256, pred1_128, pred2_128, coarseness1, coarseness2 = pl_module.forward(view1, view2)

            pred1_coarseness, pred2_coarseness = pl_module.create_coarseness_gaussians(pred1_512, pred2_512, coarseness1, coarseness2)
            # pred1_coarseness, pred2_coarseness = {}, {} 
            # for key in ["opacities", "rotations", "scales", "covariances"]:
            #     pred1_coarseness[key] = pred1_512[key].clone().requires_grad_(False)
            #     pred2_coarseness[key] = pred2_512[key].clone().requires_grad_(False)
            # pred1_coarseness['means'] = pred1_512['means'].clone().requires_grad_(False)
            # pred2_coarseness['means_in_other_view'] = pred2_512['means_in_other_view'].clone().requires_grad_(False)
            # print(f"shape sh = {pred1_512['sh'].shape}, {pred2_512['sh'].shape}")
            # print(f"shape coarseness1 = {coarseness1.shape}, coarseness2 = {coarseness2.shape}")
            # pred1_coarseness['sh'] = einops.rearrange(coarseness1, "b c h w -> b h w c 1" )
            # pred2_coarseness['sh'] = einops.rearrange(coarseness2, "b c h w -> b h w c 1" )
            coarseness_image, depth = pl_module.decoder_coarseness(batch, pred1_coarseness, pred2_coarseness, (h, w)) # gets (b, v, c, h, w)
            mask = loss_mask.calculate_loss_mask(batch)

            # Save the data
            save_dir = os.path.join(
                self.save_dir,
                f"{prefix}_epoch_{trainer.current_epoch}_batch_{batch_idx}"
            )
            log_batch_files_coarseness_pred(batch, coarseness_image, depth, mask, view1, view2, pred1_512, pred2_512, coarseness1, coarseness2, save_dir, 
                                            colors=colors, grad_coarseness=self.grad_coarseness, pl_module=pl_module)

        elif self.coarseness_predictions or self.grad_coarseness:
            # Run the batch through the model again
            _, _, h, w = batch["context"][0]["img"].shape
            view1, view2 = batch['context']
            pred1_combined, pred2_combined = pl_module.forward(view1, view2)
            print(f"pred1_combined.keys(): {pred1_combined.keys()}")
            print(f"pred2_combined.keys(): {pred2_combined.keys()}")
            color, depth = pl_module.decoder(batch, pred1_combined, pred2_combined, (h, w), fused_gaussians=True)
            
            mask = loss_mask.calculate_loss_mask(batch)

            # Save the data
            save_dir = os.path.join(
                self.save_dir,
                f"{prefix}_epoch_{trainer.current_epoch}_batch_{batch_idx}"
            )
            log_batch_files(batch, color, depth, mask, view1, view2, pred1_combined, pred2_combined, save_dir, grad_coarseness=self.grad_coarseness)

        else:
            # Run the batch through the model again
            _, _, h, w = batch["context"][0]["img"].shape
            view1, view2 = batch['context']
            pred1, pred2, pred1_lowres, pred2_lowres, _, _, = pl_module.forward(view1, view2)
            if self.coarse or self.grad_coarseness:
                pred1, pred2 = pred1_lowres, pred2_lowres
            
            color, depth = pl_module.decoder(batch, pred1, pred2, (h, w), fused_gaussians=self.grad_coarseness)
            
            mask = loss_mask.calculate_loss_mask(batch)

            # Save the data
            save_dir = os.path.join(
            # Save the data
                self.save_dir,
                f"{prefix}_epoch_{trainer.current_epoch}_batch_{batch_idx}"
            )
            log_batch_files(batch, color, depth, mask, view1, view2, pred1, pred2, save_dir, grad_coarseness=self.grad_coarseness)


def save_as_ply(pred1, pred2, save_path, as_list=False, grad_coarseness=False):
    """Save the 3D Gaussians as a point cloud in the PLY format.
    Adapted loosely from PixelSplat"""

    def construct_list_of_attributes(num_rest: int) -> list[str]:
        '''Construct a list of attributes for the PLY file format. This
        corresponds to the attributes used by online readers, such as
        https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)
        scale = scale.detach().cpu().numpy()

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()

        return quaternion, scale

    # Collect the Gaussian parameters
    # if "means_in_other_view" in pred2.keys():
    #     means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    # else:
    #     means = torch.stack([pred1["means"], pred2["means"]], dim=1)
    # covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    # harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
    # opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
    if grad_coarseness:
        means = torch.cat((pred1["means"], pred2["means_in_other_view"]), dim=1).unsqueeze(0)
        covariances = torch.cat((pred1["covariances"], pred2["covariances"]), dim=1).unsqueeze(0)
        harmonics = torch.cat((pred1["sh"], pred2["sh"]), dim=1).unsqueeze(0).squeeze(-1)
        opacities = torch.cat((pred1["opacities"], pred2["opacities"]), dim=1).unsqueeze(0)
    else:
        if "means_in_other_view" in pred2.keys():
            means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        else:
            means = torch.stack([pred1["means"], pred2["means"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
        opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
        
    # means = pred1["means"].unsqueeze(0)  # Remove the batch dimension
    # covariances = pred1["covariances"].unsqueeze(0)  # Remove the batch dimension
    # harmonics = pred1["sh"].unsqueeze(0)[..., 0]  # Only use the first harmonic
    # opacities = pred1["opacities"].unsqueeze(0)  # Remove the batch dimension

    if grad_coarseness:
        means = einops.rearrange(means[0], "view n xyz -> (view n) xyz").detach().cpu().numpy()
        covariances = einops.rearrange(covariances[0], "v n i j -> (v n) i j")
        harmonics = einops.rearrange(harmonics[0], "view n xyz -> (view n) xyz").detach().cpu().numpy()
        opacities = einops.rearrange(opacities[0], "view n xyz -> (view n) xyz").detach().cpu().numpy()
    elif not as_list:
        # Rearrange the tensors to the correct shape
        means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
        covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
        harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
        opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    else:
         # Rearrange the tensors to the correct shape
        means = einops.rearrange(means[0], "view hw xyz -> (view hw) xyz").detach().cpu().numpy()
        covariances = einops.rearrange(covariances[0], "v hw i j -> (v hw) i j")
        harmonics = einops.rearrange(harmonics[0], "view hw xyz -> (view hw) xyz").detach().cpu().numpy()
        opacities = einops.rearrange(opacities[0], "view hw xyz -> (view hw) xyz").detach().cpu().numpy()
    # Convert the covariance matrices to quaternions and scales
    rotations, scales = covariance_to_quaternion_and_scale(covariances)

    # Construct the attributes
    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Save the point cloud
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)

def save_gaussian_dpts(model, save_path):
    gaussian_dpts = {}
    for key in list(model.keys()):
        if 'gaussian_dpt' in key:
            print(key)
            gaussian_dpts[key] = model[key].clone()
            print(f"saving key {key}")
    torch.save(gaussian_dpts, save_path)
            

def save_3d(view1, view2, pred1, pred2, save_dir, as_pointcloud=True, all_points=True):
    """Save the 3D points as a point cloud or as a mesh. Adapted from DUSt3R"""

    os.makedirs(save_dir, exist_ok=True)
    batch_size = pred1["pts3d"].shape[0]
    views = [view1, view2]

    for b in range(batch_size):

        pts3d = [pred1["pts3d"][b].cpu().numpy()] + [pred2["pts3d_in_other_view"][b].cpu().numpy()]
        imgs = [einops.rearrange(view["original_img"][b], "c h w -> h w c").cpu().numpy() for view in views]
        mask = [view["valid_mask"][b].cpu().numpy() for view in views]

        # Treat all pixels as valid, because we want to render the entire viewpoint
        if all_points:
            mask = [np.ones_like(m) for m in mask]

        # Construct the scene from the 3D points as a point cloud or as a mesh
        scene = trimesh.Scene()
        if as_pointcloud:
            pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
            pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
            scene.add_geometry(pct)
            save_path = os.path.join(save_dir, f"{b}.ply")
        else:
            meshes = []
            for i in range(len(imgs)):
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)
            save_path = os.path.join(save_dir, f"{b}.glb")

        # Save the scene
        scene.export(file_obj=save_path)


@torch.no_grad()
def log_batch_files(batch, color, depth, mask, view1, view2, pred1, pred2, save_dir, should_save_3d=False, grad_coarseness=False):
    '''Save all the relevant debug files for a batch'''

    os.makedirs(save_dir, exist_ok=True)

    # Save the 3D Gaussians as a .ply file
    save_as_ply(pred1, pred2, os.path.join(save_dir, f"gaussians.ply"), grad_coarseness=True)

    # Save the 3D points as a point cloud and as a mesh (disabled)
    if should_save_3d:
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_mesh"), as_pointcloud=False)
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_pointcloud"), as_pointcloud=True)

    # Save the color, depth and valid masks for the input context images
    context_images = torch.stack([view["img"] for view in batch["context"]], dim=1)
    context_original_images = torch.stack([view["original_img"] for view in batch["context"]], dim=1)
    context_depthmaps = torch.stack([view["depthmap"] for view in batch["context"]], dim=1)
    context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)

    if "mask_highres" in pred1.keys():
        # print(f"saving mask")
        # print(f"context_images.shape: {context_original_images.shape}")
        # print(f"pred1['mask_highres'].shape: {pred1['mask_highres'].shape}")
        # print(f"pred2['mask_highres'].shape: {pred2['mask_highres'].shape}")
        context_images_lod_masked = context_original_images.clone()
        context_images_lod_masked[:,0,:,pred1['mask_highres'][0,...]] = 0.0
        context_images_lod_masked[:,1,:,pred2['mask_highres'][0,...]] = 0.0
    for b in range(min(context_images.shape[0], 4)):
        torchvision.utils.save_image(context_images[b], os.path.join(save_dir, f"sample_{b}_img_context.jpg"))
        torchvision.utils.save_image(context_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_context.jpg"))
        torchvision.utils.save_image(context_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap.jpg"), normalize=True)
        torchvision.utils.save_image(context_valid_masks[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_valid_mask_context.jpg"), normalize=True)
        if "mask_highres" in pred1.keys():
            torchvision.utils.save_image(context_images_lod_masked[b].float(), os.path.join(save_dir, f"sample_{b}_LoD_mask_context.jpg"), normalize=True)

    # Save the color and depth images for the target images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    target_depthmaps = torch.stack([view["depthmap"] for view in batch["target"]], dim=1)
    context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(target_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_target.jpg"))
        torchvision.utils.save_image(target_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap_target.jpg"), normalize=True)

    # Save the rendered images and depths
    for b in range(min(color.shape[0], 4)):
        torchvision.utils.save_image(color[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color.jpg"))
    if depth is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(depth[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_rendered_depth.jpg"), normalize=True)

    # Save the loss masks
    for b in range(min(mask.shape[0], 4)):
        torchvision.utils.save_image(mask[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_loss_mask.jpg"), normalize=True)

    # Save the masked target and rendered images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    masked_target_original_images = target_original_images * mask[..., None, :, :]
    masked_predictions = color * mask[..., None, :, :]
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(masked_target_original_images[b], os.path.join(save_dir, f"sample_{b}_masked_original_img_target.jpg"))
        torchvision.utils.save_image(masked_predictions[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_color.jpg"))


@torch.no_grad()
def log_batch_files_coarseness_pred(batch, color, depth, mask, view1, view2, pred1, pred2, coarseness1, coarseness2, save_dir, colors=None, should_save_3d=False, grad_coarseness=False, pl_module=None):
    '''Save all the relevant debug files for a batch
    color: (b, v, c, h, w) - the rendered color images for the batch
    mask: (b, v, h, w) - the loss mask for the batch
    '''
    (color_512, color_256, color_128) = colors
    os.makedirs(save_dir, exist_ok=True)

    # Save the 3D Gaussians as a .ply file
    # save_as_ply(pred1, pred2, os.path.join(save_dir, f"gaussians.ply"), grad_coarseness=grad_coarseness)

    # Save the 3D points as a point cloud and as a mesh (disabled)
    if should_save_3d:
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_mesh"), as_pointcloud=False)
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_pointcloud"), as_pointcloud=True)

    # Save the color, depth and valid masks for the input context images
    context_images = torch.stack([view["img"] for view in batch["context"]], dim=1)
    context_original_images = torch.stack([view["original_img"] for view in batch["context"]], dim=1)
    context_depthmaps = torch.stack([view["depthmap"] for view in batch["context"]], dim=1)
    context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)

    valid_1 = (pred1["conf"] > pl_module.config.loss.mast3r_confidence_threshold)
    valid_2 = (pred2["conf"] > pl_module.config.loss.mast3r_confidence_threshold)
    b=0
    _, one_hot_mask_1 = lod_utils.get_3_stage_mask(view1['original_img'][0, ...], pred1['pts3d'][b, ..., -1], valid_1[0, ...], "cuda:0", 512, 512, [0.1, 0.3], [0.2, 0.5])
    _, one_hot_mask_2 = lod_utils.get_3_stage_mask(view2['original_img'][0, ...], pred2['pts3d_in_other_view'][b, ..., -1], valid_2[0, ...], "cuda:0", 512, 512, [0.1, 0.3], [0.2, 0.5])
    # coarseness_gt.scatter_(0, torch.argmin(coarseness_mask, dim=2, keepdim=True), 1) # (b, v, 3, h, w)
    coarseness_gt_1 = one_hot_mask_1.float()
    coarseness_gt_2 = one_hot_mask_2.float()

    coarseness_gt = torch.stack((coarseness_gt_1, coarseness_gt_2), dim=0) # (b, v, 3, h, w)
    # coarseness_gt = einops.rearrange(coarseness_gt, "c h w -> h w c")
    # print(f"coarseness_gt.shape: {coarseness_gt.shape}, coarseness_gt.min(): {coarseness_gt.min()}, coarseness_gt.max(): {coarseness_gt.max()}")
    # print(f"coarseness_gt: {coarseness_gt}")
    coarseness_pred_one_hot_1 = torch.zeros_like(coarseness1)
    coarseness_pred_one_hot_2 = torch.zeros_like(coarseness2)
    # print(f"coarseness1.shape: {coarseness1.shape}, coarseness1.min(): {coarseness1.min()}, coarseness1.max(): {coarseness1.max()}")
    # coarseness_pred_one_hot[torch.argmin(coarseness1, dim=1, keepdim=True)] = 1.0
    
    argmax_1 = torch.argmax(torch.softmax(coarseness1, dim=1), dim=1, keepdim=True)
    argmax_2 = torch.argmax(torch.softmax(coarseness2, dim=1), dim=1, keepdim=True)
    # print(f"argmax shape: {argmax.shape}, argmax min: {torch.min(argmax)}, argmax max: {torch.max(argmax)}, mean: {torch.mean(argmax.float())}")
    coarseness_pred_one_hot_1.scatter_(1, argmax_1, 1.0) # (v, 3, h, w)
    coarseness_pred_one_hot_2.scatter_(1, argmax_2, 1.0) # (v, 3, h, w)
    coarseness_pred_one_hot = torch.cat((coarseness_pred_one_hot_1, coarseness_pred_one_hot_2), dim=0) # (b, v, 3, h, w)
    # print(f"coarseness_pred_one_hot: {coarseness_pred_one_hot}")


    coarseness_onehot_1 = one_hot_mask_1
    weights_1 = (512.0*512.0)/coarseness_onehot_1.sum(dim=(1,2))
    weights_normalized_1 = weights_1 / torch.linalg.norm(weights_1)
    # print(f"weights: {weights}")
    crossentropy_criterion_1 = torch.nn.CrossEntropyLoss(reduction="none", weight=weights_normalized_1) # reduction='none'
    crossentropy_loss_1 = crossentropy_criterion_1(coarseness1, coarseness_gt_1.unsqueeze(0))
    # print(f"crossentropy_loss: {crossentropy_loss.shape}")
    
    coarseness_gt_render = pl_module.calculate_loss_3stage(batch, view1, view2, colors, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False, get_gt_coarseness=True)
    # coarseness_gt_render = coarseness_gt_render - coarseness_gt_render.min() / (coarseness_gt_render.max() - coarseness_gt_render.min())
    coarseness_gt_render = coarseness_gt_render.float()
    coarseness_gt_render = einops.rearrange(coarseness_gt_render, "v h w c -> v c h w")
    mask_repeated = mask.unsqueeze(2).repeat(1, 1, 3, 1, 1)
    coarseness_gt_render[~mask_repeated[0, ...]] = 0.0
    # coarseness_gt_render = coarseness_gt_render * mask[0, ...]
    # print(f"coarseness_gt_render.shape: {coarseness_gt_render.shape}")
    # print(f"coarseness_gt_render.shape: {coarseness_gt_render}")
    # Normalize the color tensor such that the channel with the greatest value becomes 1.0 and others become 0.0
    color_one_hot = torch.zeros_like(color)
    zero_mask = (color == 0.0)
    max_indices = torch.argmax(color, dim=2, keepdim=True)  # Find the index of the max value along the channel dimension
    color_one_hot.scatter_(2, max_indices, 1.0)  # Set the max channel to 1.0 and others to 0.0
    color_one_hot[zero_mask] = 0.0  # Set the zero values to 0.0 in the one-hot encoded tensor
    # print(f"zero_mask.shape: {zero_mask.shape}, coarseness_gt_render.shape: {coarseness_gt_render.shape}")
    # print(f"mask.shape: {mask.shape}, coarseness_gt_render.min(): {coarseness_gt_render.min()}, coarseness_gt_render.max(): {coarseness_gt_render.max()}")
    if "mask_highres" in pred1.keys():
        # print(f"saving mask")
        # print(f"context_images.shape: {context_original_images.shape}")
        # print(f"pred1['mask_highres'].shape: {pred1['mask_highres'].shape}")
        # print(f"pred2['mask_highres'].shape: {pred2['mask_highres'].shape}")
        context_images_lod_masked = context_original_images.clone()
        context_images_lod_masked[:,0,:,pred1['mask_highres'][0,...]] = 0.0
        context_images_lod_masked[:,1,:,pred2['mask_highres'][0,...]] = 0.0
    for b in range(min(context_images.shape[0], 4)):
        torchvision.utils.save_image(context_images[b], os.path.join(save_dir, f"sample_{b}_img_context.jpg"))
        torchvision.utils.save_image(context_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_context.jpg"))
        torchvision.utils.save_image(context_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap.jpg"), normalize=True)
        torchvision.utils.save_image(context_valid_masks[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_valid_mask_context.jpg"), normalize=True)
        
        torchvision.utils.save_image(coarseness_pred_one_hot, os.path.join(save_dir, f"sample_{b}_coarseness1.jpg"), normalize=True)
        torchvision.utils.save_image(coarseness_gt, os.path.join(save_dir, f"sample_{b}_coarseness_mask.jpg"), normalize=True)
        torchvision.utils.save_image(crossentropy_loss_1, os.path.join(save_dir, f"sample_{b}_crossentropyloss.jpg"), normalize=True)
        torchvision.utils.save_image(coarseness_gt_render, os.path.join(save_dir, f"sample_{b}_coarseness_gt_render.jpg"), normalize=True)
        if "mask_highres" in pred1.keys():
            torchvision.utils.save_image(context_images_lod_masked[b].float(), os.path.join(save_dir, f"sample_{b}_LoD_mask_context.jpg"), normalize=True)

    # Save the color and depth images for the target images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    target_depthmaps = torch.stack([view["depthmap"] for view in batch["target"]], dim=1)
    context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(target_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_target.jpg"))
        torchvision.utils.save_image(target_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap_target.jpg"), normalize=True)

    # Save the rendered images and depths
    for b in range(min(color.shape[0], 4)):
        torchvision.utils.save_image(color[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color.jpg"))
        torchvision.utils.save_image(color_one_hot[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color_one_hot.jpg"))

        torchvision.utils.save_image(color_512[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color_512.jpg"))
        torchvision.utils.save_image(color_256[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color_256.jpg"))
        torchvision.utils.save_image(color_128[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color_128.jpg"))
    if depth is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(depth[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_rendered_depth.jpg"), normalize=True)

    # Save the loss masks
    for b in range(min(mask.shape[0], 4)):
        torchvision.utils.save_image(mask[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_loss_mask.jpg"), normalize=True)

    # Save the masked target and rendered images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    masked_target_original_images = target_original_images * mask[..., None, :, :]
    masked_predictions = color * mask[..., None, :, :]
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(masked_target_original_images[b], os.path.join(save_dir, f"sample_{b}_masked_original_img_target.jpg"))
        torchvision.utils.save_image(masked_predictions[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_color.jpg"))