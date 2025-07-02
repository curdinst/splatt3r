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
# from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.dust3r.dust3r.utils.image import load_images
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
import workspace
import main
import utils.export as export


# Load images

# load_config("config/base.yaml")

print('Loading Model')
config = workspace.load_config("configs/main.yaml")
weights_path = "checkpoints/splatt3r_pretrained.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = main.MAST3RGaussians.load_from_checkpoint(weights_path, device)

path = "/home/curdinst/repos/MASt3R-SLAM/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
file1 = "1305031452.791720.png"
file2 = "1305031454.891764.png"

files = [path + file1, path + file2]


(h, w) = (480, 640)
img_size = [h, w]
img_sizes = [img_size, img_size]
img_size = 512
images = load_images(files, img_size, verbose=True)

for img in images:
    img['true_shape'] = torch.from_numpy(img['true_shape']).to(device)
    img['img'] = img['img'].to(device)

pred1, pred2 = model(images[0], images[1])
# (shape1, shape2), (feat1, feat2), (pos1, pos2) = model.encoder._encode_symmetrized(images[0], images[1])

# dec1, dec2 = model.encoder._decoder(feat1, pos1, feat2, pos2)
# pred1 = model.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
# pred2 = model.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)
pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred1['rotations'])

export.save_as_ply(pred1, pred2, "learn_residual_true.ply")
# print(images)

# outputs = model