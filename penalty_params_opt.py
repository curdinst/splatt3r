import json
import os
import sys

import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb
import optuna

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

MAX_NUM_GAUSSIANS = 512*512*2

config = workspace.load_config(sys.argv[1], sys.argv[2:])
if os.getenv("LOCAL_RANK", '0') == '0':
    config = workspace.create_workspace(config)
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

total_res = {}
i = 0
# Testing
original_save_dir = config.save_dir
results = {}
results_avg = {}
# penalty_values = ((0.8, 0.4), (0.4, 0.2), (0.2, 0.1), (0.1, 0.05), (0.05, 0.025), (0.025, 0.0125))  # (512 penalty, 256 penalty), p512 > p256
penalty_values = ((0.0125, 0.006), (0.006, 0.003), (0.003, 0.0015))  # (512 penalty, 256 penalty), p512 > p256
# masking_configs = [[False, False]]
apply_mask = True
average_over_mask = False
abandoned = 0

def objective(trial):
    penalty_512, penalty_256 = trial.suggest_float("penalty_512", 0.05, 0.25), trial.suggest_float("penalty_256", 0.025, 0.125)
    # if penalty_512 < penalty_256:
    #     # Prune the trial if the constraint is not met
    #     # abandoned = abandoned + 1
    #     raise optuna.exceptions.TrialPruned()
    total_res = {}
    alpha_beta_pairs = ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5), (0.3, 0.3))
    # return (penalty_256-2)**2 + (penalty_512-4)**2
    # alpha_beta_pairs = [(0.5,0.5)]
    i = 0
    for alpha, beta in alpha_beta_pairs:
    # for alpha, beta in ((0.9, 0.9), (0.7, 0.7)):

        test_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=alpha,
            beta=beta,
            resolution=config.data.resolution,
            use_every_n_sample=200  # if 100 its 330 samples
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )

        new_save_dir = os.path.join(
            original_save_dir,
            f'alpha_{alpha}_beta_{beta}_penalty_512_{penalty_512}_penalty_256_{penalty_256}'
        )
        os.makedirs(new_save_dir, exist_ok=True)
        model.config.save_dir = new_save_dir

        L.seed_everything(config.seed, workers=True)

        # Training
        trainer = L.Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=[export.SaveBatchData(save_dir=config.save_dir, coarse=config.resolution < 500, grad_coarseness=config.grad_coarseness, coarseness_predictions=config.coarseness_predictions, penalty_optimisation=config.penalty_optimisation),],
            default_root_dir=config.save_dir,
            devices=config.devices,
            log_every_n_steps=10,
            strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
        )

        model.lpips_criterion = lpips.LPIPS('vgg', spatial=average_over_mask)
        model.config.loss.apply_mask = apply_mask
        model.config.loss.average_over_mask = average_over_mask
        model.config.mse_penalty_512 = penalty_512
        model.config.mse_penalty_256 = penalty_256
        res = trainer.test(model, dataloaders=data_loader_test)
        results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res
        # print(f"res: {res}")
        result = res[0]
        for key in result.keys():
            if key not in total_res.keys():
                total_res[key] = 0
            total_res[key] = total_res[key] + result[key]
        i += 1
        # Save the results
        save_path = os.path.join(original_save_dir, f'results_trial_{trial.number}_p512_{penalty_512:.2}_p256_{penalty_256:.2}.json')
        with open(save_path, 'w') as f:
            json.dump(results, f)
    for key in total_res.keys():
        total_res[key] /= i
    i = 0
    results_avg[f"p512_{penalty_512:.3}_p256_{penalty_256:.3}_average"] = total_res

    save_path = os.path.join(original_save_dir, 'results_avg.json')
    with open(save_path, 'w') as f:
        json.dump(results_avg, f)
    print(f"total res --------------- \n {total_res}")
    psnr_avg = total_res['test/mse']
    rel_num_gaussians = total_res['test/num_gaussians']/MAX_NUM_GAUSSIANS # Normalize to [0, 1]
    psnr_rescaled = psnr_avg - 25.4
    mse_avg = total_res['test/mse']
    # objective_to_minimize = - psnr_avg + 65.25 * rel_num_gaussians  # Weight factor to balance PSNR and number of Gaussians
    objective_to_minimize = mse_avg + 5e-4 * rel_num_gaussians
    return objective_to_minimize

abandoned = 0
def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=24, callbacks=[print_best_callback])

# Save the best parameters to a JSON file
best_params_path = os.path.join(original_save_dir, 'best_params.json')
# print("Number of abandoned trials due to constraints:", abandoned)

with open(best_params_path, 'w') as f:
    json.dump(study.best_params, f)
