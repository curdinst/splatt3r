import torch
from einops import rearrange, repeat

from .cuda_splatting import render_cuda
from utils.geometry import normalize_intrinsics
import os
from torchvision.utils import save_image


class DecoderSplattingCUDA(torch.nn.Module):

    def __init__(self, background_color):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
    
    def forward(self, batch, pred1, pred2, image_shape, single_map=False, fused_gaussians=False):

        base_pose = batch['context'][0]['camera_pose'] # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
        intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # Rotate the ground truth extrinsics into the coordinate system used by MAST3R
        # --i.e. in the coordinate system of the first context view, normalized by the scene scale
        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics
        # print(f"pred1['means'].shape: {pred1['means'].shape}, pred2['means_in_other_view'].shape: {pred2['means_in_other_view'].shape}")
        # print(f"pred1['covariances'].shape: {pred1['covariances'].shape}, pred2['covariances'].shape: {pred2['covariances'].shape}")
        # print(f"pred1['sh'].shape: {pred1['sh'].shape}, pred2['sh'].shape: {pred2['sh'].shape}")
        # print(f"pred1['opacities'].shape: {pred1['opacities'].shape}, pred2['opacities'].shape: {pred2['opacities'].shape}")

        if fused_gaussians:
            means = torch.cat((pred1["means"], pred2["means_in_other_view"]), dim=1).unsqueeze(0)
            covariances = torch.cat((pred1["covariances"], pred2["covariances"]), dim=1).unsqueeze(0)
            harmonics = torch.cat((pred1["sh"], pred2["sh"]), dim=1).unsqueeze(0)
            opacities = torch.cat((pred1["opacities"], pred2["opacities"]), dim=1).unsqueeze(0)
        elif not single_map:
            means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
            covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
            harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)
            opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
                    # else:
        #     means = pred1["means"]
        #     covariances = pred1["covariances"]
        #     harmonics = pred1["sh"]
        #     opacities = pred1["opacities"]

        # print(f"means.shape: {means.shape}, covariances.shape: {covariances.shape}, harmonics.shape: {harmonics.shape}, opacities.shape: {opacities.shape}")
        b, v, _, _ = extrinsics.shape
        # print(f"v: {v}, b: {b}, image_shape: {image_shape}")
        near = torch.full((b, v), 0.1, device=means.device)
        far = torch.full((b, v), 1000.0, device=means.device)
        if not fused_gaussians:
            color = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(rearrange(means, "b v h w xyz -> b (v h w) xyz"), "b g xyz -> (b v) g xyz", v=v),
                repeat(rearrange(covariances, "b v h w i j -> b (v h w) i j"), "b g i j -> (b v) g i j", v=v),
                repeat(rearrange(harmonics, "b v h w c d_sh -> b (v h w) c d_sh"), "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(rearrange(opacities, "b v h w 1 -> b (v h w)"), "b g -> (b v) g", v=v),
            )
        else:
            color = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(rearrange(means, "b v n xyz -> b (v n) xyz"), "b g xyz -> (b v) g xyz", v=v),
                repeat(rearrange(covariances, "b v n i j -> b (v n) i j"), "b g i j -> (b v) g i j", v=v),
                repeat(rearrange(harmonics, "b v n c d_sh -> b (v n) c d_sh"), "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(rearrange(opacities, "b v n 1 -> b (v n)"), "b g -> (b v) g", v=v),
            )
        # print(f"color.min: {color.min()}, color.max: {color.max()}")
        # output_dir = "/home/curdinst/repos/splatt3r/results"
        # os.makedirs(output_dir, exist_ok=True)

        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        # print(f"color.shape: {color.shape}")
        # for i in range(color.shape[0]):  # Iterate over batch
        #     for j in range(color.shape[1]):  # Iterate over views
        #         # image = rearrange(color[i, j,...], "c h w -> h w c")  # [c, h, w]
        #         image = color[i, j, ...]  # [c, h, w]
        #         print(f"image.shape: {image.shape}")
        #         save_path = os.path.join(output_dir, f"image_batch_{i}_view_{j}.png")
        #         save_image(image*256, save_path)
        return color, None