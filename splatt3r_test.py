import torch

import sys
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
sys.path.append('src/pixelsplat_src')
from dust3r.utils.image import load_images
import main
import utils.export as export

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename = "epoch=19-step=1200.ckpt"
# weights_path = hf_hub_download(repo_id=model_name, filename=filename)
weights_path = "pretrained/" + filename
RGB_PATH = "/home/curdin/repos/MASt3R-SLAM/datasets/tum/rgbd_dataset_freiburg1_room/rgb/"
file1 = RGB_PATH + "1305031910.765238.png"
file2 = RGB_PATH + "1305031911.135664.png"
filelist = [file1, file2]
image_size = 512
silent = False

model = main.MAST3RGaussians.load_from_checkpoint(weights_path, device)
print("loaded model")
imgs = load_images(filelist, size=image_size, verbose=not silent)

for img in imgs:
    img['img'] = img['img'].to(device)
    img['original_img'] = img['original_img'].to(device)
    img['true_shape'] = torch.from_numpy(img['true_shape'])

output = model(imgs[0], imgs[1])