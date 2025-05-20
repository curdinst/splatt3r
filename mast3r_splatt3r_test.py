import torch
import yaml
import pathlib
# from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
# from mast3r_slam.evaluate import save_gaussian_new_ply, save_as_ply
from src.mast3r_src.mast3r.model import *

# from mast3r_slam.dataloader import Intrinsics, load_dataset
# from mast3r_slam.frame import Frame
# from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
# from mast3r_slam.config import load_config, config, set_global_config
# from mast3r_slam.mast3r_utils import mast3r_asymmetric_inference
# from mast3r_slam.evaluate import save_gaussian_new_ply, save_as_ply
# import mast3r_slam.evaluate as eval


# Load images

# load_config("config/base.yaml")

device = "cuda"
model = load_model(device=device, model_path="/home/curdin/repos/MASt3R-SLAM/checkpoints/MASt3R_gaussians_v1.pth")
# model = load_mast3r(device=device)
model.share_memory()
print("loaded model")

# frame_1.feat, frame_1.pos, _ = model._encode_image(
#             frame_i.img, frame_i.img_true_shape




dataset_path = "datasets/tum/rgbd_dataset_freiburg1_room/"
dataset = load_dataset(dataset_path)
h, w = dataset.get_img_shape()[0]
img_size = (h, w)

img1_idx, img2_idx = 0, 30
timestamp1, img1 = dataset[img1_idx]
timestamp2, img2 = dataset[img2_idx]

image1_save_path = "logs/image" + str(img1_idx) + ".pt"
image2_save_path = "logs/image" + str(img2_idx) + ".pt"


frame1 = create_frame(img1_idx, img1, T_WC=None)
frame2 = create_frame(img2_idx, img2, T_WC=None)

# torch.save(frame1.img, image1_save_path)
# torch.save(frame2.img, image2_save_path)

# # dataset = load_dataset(args.dataset)
# # dataset.subsample(config["dataset"]["subsample"])

# calib_file = "config/intrinsics.yaml"
# with open(calib_file, "r") as f:
#             intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
# camera_intrinsics = Intrinsics.from_calib(
#             img_size,
#             intrinsics["width"],
#             intrinsics["height"],
#             intrinsics["calibration"],
#         )

# X, C, D, Q, S, R, SH, O, M, res11, res21 = mast3r_asymmetric_inference(model=model, frame_i=frame1, frame_j=frame2)

# save_dir = pathlib.Path("logs")
# # if args.save_as != "default":
# #     save_dir = save_dir / args.save_as
# save_dir.mkdir(exist_ok=True, parents=True)
# filename = "gaussians_" + str(img1_idx) + "_" + str(img2_idx) + ".ply"
# recon_file = save_dir / filename
# if recon_file.exists():
#     recon_file.unlink()

# save_as_ply(res11, res21, recon_file)

# print("predctions done")

