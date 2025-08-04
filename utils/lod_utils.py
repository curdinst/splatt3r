import torch
import torch.nn.functional as F
import einops
import utils.geometry as geometry
from matplotlib import pyplot as plt
import einops
import numpy as np
import time
from scipy.spatial.transform import Rotation
from torchvision.utils import save_image


def spatial_derivative(img, device="cuda:0"):
    """
    Compute spatial derivatives (gradients) in x and y directions
    using Sobel filters.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W)

    Returns:
        grad_x, grad_y: each of shape (C, H, W)
    """
    if img.ndim != 3:
        raise ValueError("Image must be 3D tensor (C, H, W)")

    C, H, W = img.shape

    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=device)

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    grad_x = []
    grad_y = []

    for c in range(C):
        channel = img[c:c+1, :, :].unsqueeze(0)  # shape: (1, 1, H, W)
        gx = F.conv2d(channel, sobel_x, padding=1)
        gy = F.conv2d(channel, sobel_y, padding=1)
        grad_x.append(gx.squeeze(0))
        grad_y.append(gy.squeeze(0))

    grad_x = torch.cat(grad_x, dim=0)  # (C, H, W)
    grad_y = torch.cat(grad_y, dim=0)  # (C, H, W)

    return grad_x, grad_y

def get_mask(img, depth_img, valid, device, H, W, th_rgb, th_depth):
    """
    img (torch.Tensor): Image tensor of shape (C, H, W)
    depth_img (torch.Tensor): Depth image tensor of shape (H, W)
    valid (torch.Tensor): Validity mask tensor of shape (H*W,)
    device (str): Device to perform computations on, e.g., "cuda:0" or "cpu"
    H (int): Height of the image
    W (int): Width of the image
    th_rgb (float): Threshold for RGB gradients
    th_depth (float): Threshold for depth gradients    


    Returns:
        mask_upsampled Bool tensor of shape (H, W) indicating valid pixels with original resolution
        mask_downsampled Bool tensor of shape (H//2, W//2) indicating valid pixels with downsampled resolution
    """

    grad_x, grad_y = spatial_derivative(img, device)
    grad_x, grad_y = torch.abs(grad_x), torch.abs(grad_y)
    # print(grad_x.shape, grad_y.shape)
    depth_grad_x, depth_grad_y = spatial_derivative(depth_img.unsqueeze(0), device)
    depth_grad_x, depth_grad_y = torch.abs(depth_grad_x), torch.abs(depth_grad_y)
    # print(f"depth_grad_x shape: {depth_grad_x.shape}, depth_grad_y shape: {depth_grad_y.shape}")
    # print(f"max depth_grad_x: {depth_grad_x.max()}, min depth_grad_x: {depth_grad_x.min()}")
    # print(f"max depth_grad_y: {depth_grad_y.max()}, min depth_grad_y: {depth_grad_y.min()}")

    # print("Gradient x max:", grad_x.max(), "min:", grad_x.min())
    # print("Gradient y max:", grad_y.max(), "min:", grad_y.min())

    grad_x_max, grad_x_max_indices = torch.max(grad_x, dim=0)
    grad_y_max, grad_y_max_indices = torch.max(grad_y, dim=0)

    mask_x = grad_x_max < th_rgb
    mask_y = grad_y_max < th_rgb
    depth_mask_x = depth_grad_x.squeeze(0) < th_depth
    depth_mask_y = depth_grad_y.squeeze(0) < th_depth
    # print(f"mask_x shape: {mask_x.shape}, mask_y shape: {mask_y.shape}")
    # print(f"depth_mask_x shape: {depth_mask_x.shape}, depth_mask_y shape: {depth_mask_y.shape}")
    if len(valid.shape) == 1:
        valid_rearranged = einops.rearrange(valid, "(h w) -> h w", h=H, w=W)
    else:
        valid_rearranged = valid
    # print(f"valid_rearranged shape: {valid_rearranged.shape}")
    mask = mask_x & mask_y & depth_mask_x & depth_mask_y & valid_rearranged
    # print("Mask shape:", mask.shape)
    # print(f"mask.sum(): {mask.sum()}, mask.numel(): {mask.numel()}, mask.sum()/mask.numel(): {mask.sum()/mask.numel()}")
    # print(H,W, "Downsampled mask shape:", mask.shape)

    H_mask, W_mask = H // 2, W // 2
    # mask_downsampled = F.upsample(mask.float().unsqueeze(0), size=(H_mask,W_mask), mode="bilinear")
    mask_downsampled = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(H_mask, W_mask), mode="bilinear")
    # print("Downsampled mask shape:", mask_downsampled.shape)
    mask_downsampled = (mask_downsampled > 0.9)
    # print("Downsampled mask sum:", mask_downsampled.sum(), "numel:", mask_downsampled.numel(), "ratio:", mask_downsampled.sum()/mask_downsampled.numel())
    mask_upsampled = F.interpolate(mask_downsampled.float(), size=(H, W), mode="nearest").squeeze(0).squeeze(0).bool()
    # print("mask_downsampled.shape :", mask_downsampled.shape, "mask_upsampled.shape:", mask_upsampled.shape)
    return mask_downsampled.squeeze(0).squeeze(0), ~mask_upsampled.squeeze(0).squeeze(0)  # Returns (H//2, W//2) and (H, W) masks

def get_3_stage_mask(img, depth_img, valid, device, H, W, th_rgb, th_depth):
    """
    img (torch.Tensor): Image tensor of shape (C, H, W)
    depth_img (torch.Tensor): Depth image tensor of shape (H, W)
    valid (torch.Tensor): Validity mask tensor of shape (H*W,)
    device (str): Device to perform computations on, e.g., "cuda:0" or "cpu"
    H (int): Height of the image
    W (int): Width of the image
    th_rgb (float, float): Threshold for RGB gradients
    th_depth (float, float): Threshold for depth gradients    


    Returns:
        mask_upsampled Bool tensor of shape (H, W) indicating valid pixels with original resolution
        mask_downsampled Bool tensor of shape (H//2, W//2) indicating valid pixels with downsampled resolution
    """

    grad_x, grad_y = spatial_derivative(img, device)
    grad_x, grad_y = torch.abs(grad_x), torch.abs(grad_y)
    # print(grad_x.shape, grad_y.shape)
    depth_grad_x, depth_grad_y = spatial_derivative(depth_img.unsqueeze(0), device)
    depth_grad_x, depth_grad_y = torch.abs(depth_grad_x), torch.abs(depth_grad_y)
    # print(f"depth_grad_x shape: {depth_grad_x.shape}, depth_grad_y shape: {depth_grad_y.shape}")
    # print(f"max depth_grad_x: {depth_grad_x.max()}, min depth_grad_x: {depth_grad_x.min()}")
    # print(f"max depth_grad_y: {depth_grad_y.max()}, min depth_grad_y: {depth_grad_y.min()}")

    # print("Gradient x max:", grad_x.max(), "min:", grad_x.min())
    # print("Gradient y max:", grad_y.max(), "min:", grad_y.min())

    grad_x_max, grad_x_max_indices = torch.max(grad_x, dim=0)
    grad_y_max, grad_y_max_indices = torch.max(grad_y, dim=0)
    
    th_rgb_low, th_rgb_high = th_rgb
    th_depth_low, th_depth_high = th_depth

    mask_x_high = grad_x_max < th_rgb_high
    mask_y_high = grad_y_max < th_rgb_high
    depth_mask_x = depth_grad_x.squeeze(0) < th_depth_high
    depth_mask_y = depth_grad_y.squeeze(0) < th_depth_high

    mask_x_low = grad_x_max < th_rgb_low
    mask_y_low = grad_y_max < th_rgb_low
    depth_mask_x_low = depth_grad_x.squeeze(0) < th_depth_low
    depth_mask_y_low = depth_grad_y.squeeze(0) < th_depth_low

    # print(f"mask_x shape: {mask_x.shape}, mask_y shape: {mask_y.shape}")
    # print(f"depth_mask_x shape: {depth_mask_x.shape}, depth_mask_y shape: {depth_mask_y.shape}")
    if len(valid.shape) == 1:
        valid_rearranged = einops.rearrange(valid, "(h w) -> h w", h=H, w=W)
    else:
        valid_rearranged = valid
    # print(f"valid_rearranged shape: {valid_rearranged.shape}")
    mask_high = mask_x_high & mask_y_high & depth_mask_x & depth_mask_y & valid_rearranged

    mask_low = mask_x_low & mask_y_low & depth_mask_x_low & depth_mask_y_low & valid_rearranged

    mask_128 = mask_low
    mask_256 = mask_high & ~mask_low
    mask_512 = ~mask_high & ~mask_low
    print(f"mask_128.shape : {mask_128.shape}, mask_256.shape: {mask_256.shape}, mask_512.shape: {mask_512.shape}")
    corseness_classes = torch.zeros((1, H, W), device=device, dtype=torch.long) # (1, H, W)
    corseness_classes[0, mask_128] = 2
    corseness_classes[0, mask_256] = 1
    corseness_classes[0, mask_512] = 0

    one_hot_mask = torch.stack([mask_512, mask_256, mask_128], dim=0)  # (3, H, W)
    return corseness_classes, one_hot_mask # Returns (1, H, W) mask with 3 stages: 512, 256, 128 AND a one_hot_mask with (3, H, W)

    # coarseness_image = torch.stack([mask_512, mask_256, mask_128], dim=0)  # (3, H, W)
    # return coarseness_image # Returns (3, H, W) mask with 3 stages: 512, 256, 128

    # print("Mask shape:", mask.shape)
    # print(f"mask.sum(): {mask.sum()}, mask.numel(): {mask.numel()}, mask.sum()/mask.numel(): {mask.sum()/mask.numel()}")
    # print(H,W, "Downsampled mask shape:", mask.shape)

    # H_mask, W_mask = H // 2, W // 2
    # # mask_downsampled = F.upsample(mask.float().unsqueeze(0), size=(H_mask,W_mask), mode="bilinear")
    # mask_downsampled = F.interpolate(mask_high.unsqueeze(0).unsqueeze(0).float(), size=(H_mask, W_mask), mode="bilinear")
    # # print("Downsampled mask shape:", mask_downsampled.shape)
    # mask_downsampled = (mask_downsampled > 0.9)
    # # print("Downsampled mask sum:", mask_downsampled.sum(), "numel:", mask_downsampled.numel(), "ratio:", mask_downsampled.sum()/mask_downsampled.numel())
    # mask_upsampled = F.interpolate(mask_downsampled.float(), size=(H, W), mode="nearest").squeeze(0).squeeze(0).bool()
    # # print("mask_downsampled.shape :", mask_downsampled.shape, "mask_upsampled.shape:", mask_upsampled.shape)
    # return mask_downsampled.squeeze(0).squeeze(0), ~mask_upsampled.squeeze(0).squeeze(0)  # Returns (H//2, W//2) and (H, W) masks

def apply_mask_to_gaussians(pred, pred_lowres, mask, mask_lowres):
    """
    Apply a mask to the predicted Gaussian parameters.

    Args:
        pred (dict): containing Gaussian parameters, 
        pts3d, ([b, 512, 512, 3])
        conf, ([b, 512, 512])
        desc, ([b, 512, 512, 24])
        desc_conf, ([b, 512, 512])
        scales, ([b, 512, 512, 3])
        rotations, ([b, 512, 512, 4])
        sh, ([b, 512, 512, 3, 1])
        opacities, ([b, 512, 512, 1])
        means, ([b, 512, 512, 3])
        covariances, ([b, 512, 512, 3, 3])
        (torch.Tensor): Mask tensor of shape (b, H, W).

    Returns:
        tuple: pred_combined containging ombined Gaussians
    """

    pred_combined = {}
    for key in pred.keys():
        if key not in ["means", "opacities", "sh", "rotations", "scales", "covariances"]:
            continue
        # print(f"pred[key].shape: {pred[key].shape}, pred_lowres[key].shape: {pred_lowres[key].shape}")
        # print(f"mask shape: {mask.shape}, mask_lowres shape: {mask_lowres.shape}")
        # pred_key = einops.rearrange(pred[key], "b h w ... -> b (h w) ...")
        # pred_lowres_key = einops.rearrange(pred_lowres[key], "b h w ... -> b (h w) ...")
        # mask = einops.rearrange(mask, "b h w -> b (h w)")
        # mask_lowres = einops.rearrange(mask_lowres, "b h w -> b (h w)")
        # pred_masked = torch.index_select(pred_key, 1, mask)
        # pred_lowres_masked = torch.index_select(pred_lowres_key, 1, mask_lowres)
        # print(f"pred_masked.shape: {pred_masked.shape}, pred_lowres_masked.shape: {pred_lowres_masked.shape}")
        # pred_combined[key] = torch.cat([pred[key][mask,...], pred_lowres[key][mask_lowres,...]], dim=1)
        # print(pred_combined[key].shape, "for key:", key)
        tensor_list = []
        for b in range(pred["means"].shape[0]):
            # print(f"fusing key, {key}, pred[key].shape: {pred[key].shape}, pred_lowres[key].shape: {pred_lowres[key].shape}")
            tensor_list.append(torch.cat([pred[key][b,mask[b,...],...], pred_lowres[key][b,mask_lowres[b,...],:]], dim=0))
        pred_combined[key] = torch.stack(tensor_list, dim=0)
        # print(f"-- pred_combined[key].shape: {pred_combined[key].shape}, for key: {key}")
    return pred_combined


def fuse_gaussians(gaussians_in, valid, img, depth, img_size, config, device):
    (H, W) = img_size
    (means, sh, opacities, scales, rotations) = gaussians_in
    covariances = geometry.build_covariance(scales, rotations)
    # print(f"sh shape 1: {sh.shape}, opacities shape: {opacities.shape}, means shape: {means.shape}, scales shape: {scales.shape}, rotations shape: {rotations.shape}")

    spherical_harmonics = einops.rearrange(sh, "(h w) c d -> (c d) h w", h=H, w=W)
    opacities = einops.rearrange(opacities, "(h w) c -> c h w", h=H, w=W)
    means = einops.rearrange(means, "(h w) c -> c h w", h=H, w=W)
    covariances = einops.rearrange(covariances, "(h w) x y -> x y h w", h=H, w=W)
    scales = einops.rearrange(scales, "(h w) c -> c h w", h=H, w=W)
    rotations = einops.rearrange(rotations, "(h w) c -> c h w", h=H, w=W)
    depth = einops.rearrange(depth, "(h w) -> h w", h=H, w=W)
    th_rgb, th_depth = config["gaussians"]["gaussian_fusion_params"]["rgb_threshold"], config["gaussians"]["gaussian_fusion_params"]["depth_threshold"]
    mask_downsampled, mask_upsampled = get_mask(img, depth, valid, device, H, W, th_rgb, th_depth)
    # print("Mask shape:", mask_downsampled.shape)
    # print(mask_downsampled)
    num_fused = mask_downsampled.sum()
    #upsample mask again to take gaussians that are not in the mask
    indices = torch.nonzero(mask_downsampled, as_tuple=False)
    # print(f"Indices shape: {indices.shape}, num_fused: {num_fused}")
    # Get u, v coordinates from indices
    u = indices[:, 0] * 2
    v = indices[:, 1] * 2
    # print(f"u shape: {u.shape}, v shape: {v.shape}")
# print(f"max u {u.max()}, max v {v.max()}")

    fused_means = (means[:, u, v] + means[:, u, v+1] + means[:, u+1, v] + means[:, u+1, v+1]) / 4.0
    mean_offset1 = means[:, u, v] - fused_means
    mean_offset2 = means[:, u, v+1] - fused_means
    mean_offset3 = means[:, u+1, v] - fused_means
    mean_offset4 = means[:, u+1, v+1] - fused_means

    fused_means = (means[:, u, v] + means[:, u, v+1] + means[:, u+1, v] + means[:, u+1, v+1]) / 4.0

    # print(f"mean_offset1 shape: {mean_offset1.shape}")
    # print(f"mean_offset1 max: {mean_offset1.max()}, min: {mean_offset1.min()}, mean: {mean_offset1.mean(axis=1)}") 
    # print(F"mean_offset2 max: {mean_offset2.max()}, min: {mean_offset2.min()}, mean: {mean_offset2.mean(axis=1)}") 
    # print(f"mean_offset3 max: {mean_offset3.max()}, min: {mean_offset3.min()}, mean: {mean_offset3.mean(axis=1)}") 
    # print(f"mean_offset4 max: {mean_offset4.max()}, min: {mean_offset4.min()}, mean: {mean_offset4.mean(axis=1)}")

    matrix1 = torch.einsum('ji,ki->jki', mean_offset1, mean_offset1)
    matrix2 = torch.einsum('ji,ki->jki', mean_offset2, mean_offset2)
    matrix3 = torch.einsum('ji,ki->jki', mean_offset3, mean_offset3)
    matrix4 = torch.einsum('ji,ki->jki', mean_offset4, mean_offset4)
    # print(f"meanoffset1: {mean_offset1[:, 0]}")
    # print(f"matrix1: {matrix1[0, ...]}")
    # print(f"matrix1 shape: {matrix1.shape}")
    # print(f"mean_offset1 mean {mean_offset1.mean(axis=1)}, max {mean_offset1.max()}, min {mean_offset1.min()}")
    fused_covariances = (0.25 * covariances[..., u, v] + matrix1
                        + 0.25 * covariances[..., u, v+1] + matrix2
                        + 0.25 * covariances[..., u+1, v] + matrix3
                        + 0.25 * covariances[..., u+1, v+1] + matrix4)
    fused_opacities = (opacities[:, u, v] + opacities[:, u, v+1] + opacities[:, u+1, v] + opacities[:, u+1, v+1]) / 4.0
    fused_sh = (spherical_harmonics[:, u, v] + spherical_harmonics[:, u, v+1] + spherical_harmonics[:, u+1, v] + spherical_harmonics[:, u+1, v+1]) / 4.0
    # print("Fused means shape:", fused_means.shape)
    # print(f"fused_covariances shape: {fused_covariances.shape}")

    # fused_means = einops.rearrange(fused_means, "c n -> n c")
    # fused_opacities = einops.rearrange(fused_opacities, "c n-> n c")
    # fused_sh = einops.rearrange(fused_sh, "c n -> n c")
    fused_covariances = einops.rearrange(fused_covariances, "x y n -> n x y")
    fused_rotations, fused_scales = geometry.covariance_to_quaternion_and_scale(fused_covariances)
    fused_rotations = einops.rearrange(fused_rotations, "n c -> c n")
    fused_scales = einops.rearrange(fused_scales, "n c -> c n")
    # print(f"fused_scales shape: {fused_scales.shape}")
    valid_rearranged = einops.rearrange(valid, "(h w) -> h w", h=H, w=W)
    original_valid = ~mask_upsampled & valid_rearranged
    num_gaussians_original = valid.sum()
    
    # original_means = means[:, original_valid]
    # original_opacities = opacities[:, original_valid]
    # original_covariances = covariances[..., original_valid]
    # original_sh = spherical_harmonics[:, original_valid]
    # original_scales = scales[:, original_valid]
    # original_rotations = rotations[:, original_valid]
    means[..., u,v] = fused_means
    spherical_harmonics[..., u,v] = fused_sh
    opacities[..., u,v] = fused_opacities
    scales[..., u,v] = fused_scales
    rotations[..., u,v] = fused_rotations
    new_valid = original_valid.clone()
    new_valid[u,v] = True
    # print(f"sh shape: {spherical_harmonics.shape}, opacities shape: {opacities.shape}, means shape: {means.shape}, scales shape: {scales.shape}, rotations shape: {rotations.shape}")
    spherical_harmonics = einops.rearrange(spherical_harmonics, "(c d) h w-> (h w) c d", c=3, d=1)
    # print(f"sh shape: {spherical_harmonics.shape}, opacities shape: {opacities.shape}, means shape: {means.shape}, scales shape: {scales.shape}, rotations shape: {rotations.shape}")
    opacities = einops.rearrange(opacities, "c h w -> (h w) c", h=H, w=W)
    means = einops.rearrange(means, "c h w -> (h w) c", h=H, w=W)
    scales = einops.rearrange(scales, "c h w -> (h w) c", h=H, w=W)
    rotations = einops.rearrange(rotations, "c h w -> (h w) c", h=H, w=W)
    new_valid = einops.rearrange(new_valid, "h w -> (h w)", h=H, w=W)

    # original_means = einops.rearrange(original_means, "c n -> n c")
    # original_opacities = einops.rearrange(original_opacities, "c n-> n c")
    # original_sh = einops.rearrange(original_sh, "c n -> n c")
    # original_scales = einops.rearrange(original_scales, "c n -> n c")
    # original_rotations = einops.rearrange(original_rotations, "c n -> n c")

    # reduced_means = torch.cat((fused_means, original_means), dim=0)
    # reduced_opacities = torch.cat((fused_opacities, original_opacities), dim=0)
    # reduced_sh = torch.cat((fused_sh, original_sh), dim=0)
    # reduced_scales = torch.cat((fused_scales, original_scales), dim=0)
    # reduced_rotations = torch.cat((fused_rotations, original_rotations), dim=0)
    num_gaussians = new_valid.sum()
    print(f"num gaussians: {num_gaussians}, num gaussians original: {num_gaussians_original}")
    # reduced_sh = einops.rearrange(reduced_sh, "n c -> n c 1")

    return(means, spherical_harmonics, opacities, scales, rotations), new_valid


