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
from main import MAST3RGaussians

def run_experiment(config):

    # Set the seed
    L.seed_everything(config.seed, workers=True)

    if config.resolution < 500:
        print(f"Training Coarse Head")

    # Set up loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=config.name
        )
        loggers.append(csv_logger)

    profiler = None

    # Model
    print('Loading Model')
    config.grad_gaussian_dpt = False
    config.grad_gaussian_lowres_dpt = False
    if config.use_pretrained:
        model = MAST3RGaussians.load_from_checkpoint(checkpoint_path=config.pretrained_mast3r_path, device='cuda:0', config=config)
    else:
        model = MAST3RGaussians(config)


    # Training Datasets
    print(f'Building Datasets')
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

        # masking_configs = ((True, False), (True, True))
        masking_configs = [[False, False]]
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
                callbacks=[export.SaveBatchData(save_dir=config.save_dir, coarse=config.resolution < 500, lod=config.use_lod),],
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
