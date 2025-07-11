# import json
# import os
# import sys

# import einops
# import lightning as L
# import lpips
# import omegaconf
import torch
# import wandb

# # Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
# sys.path.append('src/pixelsplat_src')
# sys.path.append('src/mast3r_src')
# sys.path.append('src/mast3r_src/dust3r')
# from thirdparty.splatt3r.src.mast3r_src.dust3r.dust3r.losses import L21
# from thirdparty.splatt3r.src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
# import thirdparty.splatt3r.data.scannetpp.scannetpp as scannetpp
# import thirdparty.splatt3r.src.mast3r_src.mast3r.model as mast3r_model
# import thirdparty.splatt3r.src.pixelsplat_src.benchmarker as benchmarker
# import thirdparty.splatt3r.src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
# import thirdparty.splatt3r.utils.compute_ssim as compute_ssim
# import thirdparty.splatt3r.utils.export as export
# import thirdparty.splatt3r.utils.geometry as geometry
# import thirdparty.splatt3r.utils.loss_mask as loss_mask
# import thirdparty.splatt3r.utils.sh_utils as sh_utils
# import workspace


# mast3r = torch.load('checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
# splatt3r = torch.load('checkpoints/splatt3r.ckpt')
# print(mast3r.keys())
# print(mast3r['args'])
# print("---------------------------------------------------------------")
# print(splatt3r.keys())

# mast3r_gaussians = mast3r.copy()
# for key in splatt3r['state_dict'].keys():
#     key_modified = key.replace('encoder.', '')
#     mast3r_gaussians['model'][key_modified] = splatt3r['state_dict'][key]
# mast3r_gaussians['model'] = splatt3r['state_dict']


# MASt3R_gaussians_v1 = torch.load('checkpoints/MASt3R_gaussians_v1.pth', map_location='cpu')

# ADD Gaussian DPT for low resolutions ===========================================================================
# filename = "epoch=19-step=1200.ckpt"
# MASt3R_gaussians_v1 = torch.load('pretrained/'+filename, map_location='cpu')
# MASt3R_gaussians_v1_keys = MASt3R_gaussians_v1['state_dict'].keys()

# splatt3r_lowres = MASt3R_gaussians_v1.copy()

# for key in list(MASt3R_gaussians_v1_keys):
#     if 'gaussian_dpt' in key:
#         print(key)
#         key_modified = key.replace('gaussian_dpt', 'gaussian_dpt_lowres')
#         splatt3r_lowres['state_dict'][key_modified] = MASt3R_gaussians_v1['state_dict'][key]

# torch.save(splatt3r_lowres, 'pretrained/splatt3r_lowres.pth')
#===================================================================================================================
filename = "epoch=19-step=1200.ckpt"
splatt3r_lowres = torch.load('pretrained/splatt3r_lowres.pth')
splatt3r = torch.load('pretrained/' + filename)

print(len(splatt3r_lowres['state_dict'].items()))
print(len(splatt3r['state_dict'].items()))

# print(MASt3R_gaussians_v1['model']['downstream_head1.dpt.act_postprocess.0.0.weight'])


# torch.save(mast3r_gaussians, 'checkpoints/MASt3R_gaussians_v1.pth')
