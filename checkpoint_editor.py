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


def clone_head():
    # ADD Gaussian DPT for low resolutions ===========================================================================
    filename = "splatt3r_coarse.ckpt"
    MASt3R_gaussians_v1 = torch.load('pretrained/'+filename, map_location='cpu')
    MASt3R_gaussians_v1_keys = MASt3R_gaussians_v1['state_dict'].keys()

    splatt3r_lowres = MASt3R_gaussians_v1.copy()

    for key in list(MASt3R_gaussians_v1_keys):
        if 'gaussian_dpt_lowres' in key:
            print(key)
            key_modified = key.replace('lowres', '256')
            # splatt3r_lowres['state_dict'][key_modified] = MASt3R_gaussians_v1['state_dict'][key].clone()
            # splatt3r_lowres['state_dict'][key_modified].requires_grad = True
            key_modified_128 = key_modified.replace('256', '128')
            splatt3r_lowres['state_dict'][key_modified_128] = MASt3R_gaussians_v1['state_dict'][key].clone()
            splatt3r_lowres['state_dict'][key_modified] = MASt3R_gaussians_v1['state_dict'].pop(key)


            # splatt3r_lowres['state_dict'][key_modified_128].requires_grad = True
            print(f"{key_modified} requires_grad: {splatt3r_lowres['state_dict'][key_modified].requires_grad}")
            print(f"{key_modified_128} requires_grad: {splatt3r_lowres['state_dict'][key_modified_128].requires_grad}")

    # torch.save(splatt3r_lowres, 'pretrained/splatt3r_3stage.ckpt')
    #===================================================================================================================

def save_dpt_params(filename, savepath=None):
    modelpath = savepath + filename
    model = torch.load(modelpath, map_location='cpu')
    dpt_params = {}
    print(model.keys())
    # return
    i = 0
    for key, value in model['state_dict'].items():
        print(key)
        i += 1
        # if 'gaussian_dpt' in key:
        #     dpt_params[key] = value
        #     print(f"Extracted {key} with shape {value.shape}")
    print(f"Total DPT parameters extracted: {i}")
    # torch.save(dpt_params, modelpath.replace('.ckpt', '_dpt_params.pth'))

# clone_head()

CHECKPOINTS_DIR = "/mnt/buzz_newhd/home/v4rl/splatt3r/checkpoints/keep/"

# model_name = "splatt3r_coarse=0_epoch=09_batch2"
# model_name = "25-08-06-12-31-28_epoch=04_step=13620"
# # model_name = "25-08-07-11-44-51_epoch=15_step=65376"

# # filename = model_name + ".ckpt"
# # save_dpt_params(filename, savepath=CHECKPOINTS_DIR)

# filename = "splatt3r_3stage_freq_pred.ckpt"

# wrong_trained_model = torch.load(CHECKPOINTS_DIR + filename, map_location='cpu')
# new_model = wrong_trained_model.copy()
# for key, value in wrong_trained_model['state_dict'].items():
#     if 'gaussian_dpt_256' in key:
#         key_512 = key.replace('gaussian_dpt_256', 'gaussian_dpt')
#         new_model['state_dict'][key] = wrong_trained_model['state_dict'][key_512].clone()
#         new_model['state_dict'][key].requires_grad = True
#     if 'gaussian_dpt_128' in key:
#         key_512 = key.replace('gaussian_dpt_128', 'gaussian_dpt')
#         new_model['state_dict'][key] = wrong_trained_model['state_dict'][key_512].clone()
#         new_model['state_dict'][key].requires_grad = True
#     # if 'coarseness' in key:
#     #     new_model['state_dict'].pop(key)

# torch.save(new_model, CHECKPOINTS_DIR + 'splatt3r_3stage_base.ckpt')



# checkpoint_working = torch.load(CHECKPOINTS_DIR[:-5] + '25-08-29-02-41-17_epoch=10_step=22473.ckpt')
# checkpoint_crashing = torch.load(CHECKPOINTS_DIR + 'splatt3r_3stage_base.ckpt')

# # Compare the keys in the state_dict
# # print("Working keys:", checkpoint_working['state_dict'].keys(), "\n")
# # print("Crashing keys:", checkpoint_crashing['state_dict'].keys(), "\n")
# for key in checkpoint_working['state_dict'].keys():
#     if key not in checkpoint_crashing['state_dict']:
#         print(f"Key {key} is missing in crashing checkpoint")
#     print(f"working: {checkpoint_working['state_dict'][key].shape}")
#     print(f"crashin: {checkpoint_crashing['state_dict'][key].shape}")
#     if checkpoint_working['state_dict'][key].shape != checkpoint_crashing['state_dict'][key].shape:
#         print(f"Key {key} has different shapes: {checkpoint_working['state_dict'][key].shape} vs {checkpoint_crashing['state_dict'][key].shape}")

# for key in checkpoint_crashing['state_dict'].keys():
#     if key not in checkpoint_working['state_dict']:
#         print(f"Key {key} is missing in working checkpoint")
#     # else:
        
# You can also compare hyperparameters
# print("Working hparams:", checkpoint_working['hyper_parameters'], "\n")
# print("Crashing hparams:", checkpoint_crashing['hyper_parameters'], "\n")

# filename = "epoch=19-step=1200.ckpt"
# splatt3r_lowres = torch.load('pretrained/splatt3r_lowres.pth')
# splatt3r = torch.load('pretrained/' + filename)

# print(len(splatt3r_lowres['state_dict'].items()))
# print(len(splatt3r['state_dict'].items()))

# print(MASt3R_gaussians_v1['model']['downstream_head1.dpt.act_postprocess.0.0.weight'])


# torch.save(mast3r_gaussians, 'checkpoints/MASt3R_gaussians_v1.pth')

# exit()
# ---------------------------------------------------------------------------------------------------------------
filename = "splatt3r_use_trained.ckpt"
# # splatt3r_lowres = torch.load('pretrained/splatt3r_lowres.pth')
# # splatt3r = torch.load('checkpoints/' + "splatt3r_no_coarseness.ckpt")
splatt3r = torch.load('checkpoints/keep/' + filename)

# splatt3r_trained = torch.load('checkpoints/keep/' + '25-08-26-14-38-58_epoch=17_step=73548.ckpt')
splatt3r_trained = torch.load('checkpoints/keep/' + '25-09-02-19-20-30_epoch=14_step=40860_train_coarsness.ckpt')

# # print(len(splatt3r_lowres['state_dict'].items()))
# # print(len(splatt3r['state_dict'].keys()))
# splatt3r_no_coarseness = splatt3r.copy()
# # print(splatt3r['model']['downstream_head1.dpt.act_postprocess.0.0.weight'])
for key in list(splatt3r['state_dict'].keys()):
    if  'coarseness' in key or 'gaussian' in key:
        print(f"Key: {key}")
        # splatt3r['state_dict'].pop(key)
        print(f"{key} requires_grad: {splatt3r['state_dict'][key].requires_grad}")
        requires_grad = splatt3r['state_dict'][key].requires_grad
        splatt3r['state_dict'][key] = splatt3r_trained['state_dict'][key].clone()
        print(f"{key} requires_grad: {splatt3r['state_dict'][key].requires_grad}")
        splatt3r['state_dict'][key].requires_grad = True if 'gaussian' in key else requires_grad
        # print(splatt3r['state_dict'][key])
        # # splatt3r['state_dict'][key].requires_grad = False


torch.save(splatt3r, 'checkpoints/keep/splatt3r_coarseness_opt_params.ckpt')
# -----------------------------------------------------------------------------------------------------------------


# import torch
# from collections import OrderedDict

# def compare_checkpoints(checkpoint_path1: str, checkpoint_path2: str):
#     """
#     Compares two PyTorch Lightning checkpoint files to detect architectural and hyperparameter differences.

#     Args:
#         checkpoint_path1 (str): Path to the first checkpoint file (.ckpt).
#         checkpoint_path2 (str): Path to the second checkpoint file (.ckpt).
#     """
#     try:
#         # Load the checkpoints
#         print(f"Loading checkpoint from: {checkpoint_path1}")
#         ckpt1 = torch.load(checkpoint_path1, map_location='cpu')

#         print(f"Loading checkpoint from: {checkpoint_path2}")
#         ckpt2 = torch.load(checkpoint_path2, map_location='cpu')
#     except FileNotFoundError as e:
#         print(f"Error: One of the checkpoint files was not found. {e}")
#         return
#     except Exception as e:
#         print(f"An error occurred while loading checkpoints: {e}")
#         return

#     # --- Step 1: Compare State Dictionaries (Model Architecture) ---
#     print("\n" + "="*50)
#     print("      COMPARING MODEL STATE DICTIONARIES (KEYS)")
#     print("="*50)

#     state_dict1 = ckpt1.get('state_dict', OrderedDict())
#     state_dict2 = ckpt2.get('state_dict', OrderedDict())

#     keys1 = set(state_dict1.keys())
#     keys2 = set(state_dict2.keys())

#     # Find keys that are unique to each checkpoint
#     unique_keys_ckpt1 = keys1 - keys2
#     unique_keys_ckpt2 = keys2 - keys1

#     if unique_keys_ckpt1:
#         print("\nKeys unique to Checkpoint 1:")
#         for key in unique_keys_ckpt1:
#             print(f"  - {key}")
    
#     if unique_keys_ckpt2:
#         print("\nKeys unique to Checkpoint 2:")
#         for key in unique_keys_ckpt2:
#             print(f"  - {key}")

#     if not unique_keys_ckpt1 and not unique_keys_ckpt2:
#         print("\n✅ The state dictionaries have the same set of keys.")
#     else:
#         print("\n🚨 WARNING: The models have different architectures. This is the likely cause of your saving error.")
        
#     # Compare the shapes of shared keys
#     print("\n" + "="*50)
#     print("  COMPARING TENSOR SHAPES FOR COMMON KEYS")
#     print("="*50)
    
#     common_keys = keys1.intersection(keys2)
#     shape_mismatches = []
#     for key in common_keys:
#         shape1 = state_dict1[key].shape
#         shape2 = state_dict2[key].shape
#         # if state_dict1[key].requires_grad != state_dict2[key].requires_grad:
#         #     print(f"key1: {key} : {state_dict1[key].requires_grad}, key2: {state_dict2[key].requires_grad}")
#         # print(f"dict1 device: {state_dict1[key].device}, dict2 device: {state_dict2[key].device}")
#         if state_dict1[key].device != state_dict2[key].device:
#             print(f"key1: {key} : {state_dict1[key].device}, key2: {state_dict2[key].device}")
#         if state_dict1[key].dtype != state_dict2[key].dtype:
#             print(f"dtype: key1: {key} : {state_dict1[key].dtype}, key2: {state_dict2[key].dtype}")

#         if shape1 != shape2:
#             shape_mismatches.append((key, shape1, shape2))

#     if shape_mismatches:
#         print("\n🚨 The following layers have mismatched tensor shapes:")
#         for key, shape1, shape2 in shape_mismatches:
#             print(f"  - Key: {key}")
#             print(f"    - Checkpoint 1 shape: {shape1}")
#             print(f"    - Checkpoint 2 shape: {shape2}")
#     else:
#         print("\n✅ All shared layers have matching tensor shapes.")

#     # --- Step 2: Compare Hyperparameters ---
#     print("\n" + "="*50)
#     print("      COMPARING HYPERPARAMETERS")
#     print("="*50)

#     hparams1 = ckpt1.get('hyper_parameters', {})
#     hparams2 = ckpt2.get('hyper_parameters', {})

#     hparams_keys1 = set(hparams1.keys())
#     hparams_keys2 = set(hparams2.keys())

#     unique_hparams1 = hparams_keys1 - hparams_keys2
#     unique_hparams2 = hparams_keys2 - hparams_keys1
    
#     if unique_hparams1:
#         print("\nHyperparameters unique to Checkpoint 1:")
#         for key in unique_hparams1:
#             print(f"  - {key}: {hparams1.get(key)}")
            
#     if unique_hparams2:
#         print("\nHyperparameters unique to Checkpoint 2:")
#         for key in unique_hparams2:
#             print(f"  - {key}: {hparams2.get(key)}")
            
#     common_hparams = hparams_keys1.intersection(hparams_keys2)
    
#     value_mismatches = []
#     for key in common_hparams:
#         if hparams1.get(key) != hparams2.get(key):
#             value_mismatches.append((key, hparams1.get(key), hparams2.get(key)))
            
#     if value_mismatches:
#         print("\n🚨 The following common hyperparameters have mismatched values:")
#         for key, val1, val2 in value_mismatches:
#             print(f"  - Key: {key}")
#             print(f"    - Checkpoint 1 value: {val1}")
#             print(f"    - Checkpoint 2 value: {val2}")
#             for key2 in val1.keys():
#                 if key2 not in val2.keys():
#                     print(f"  - Key: {key2} is unique to one checkpoint.")
#                 elif val1[key2] != val2[key2]:
#                     print(f"    - Mismatched sub-key: {key2}")
#                     print(f"      - Checkpoint 1 value: {val1[key2]}")
#                     print(f"      - Checkpoint 2 value: {val2[key2]}")
    
#     if not unique_hparams1 and not unique_hparams2 and not value_mismatches:
#         print("\n✅ All hyperparameters are identical.")
#     else:
#         print("\n🚨 WARNING: Hyperparameters are different between the checkpoints.")

#     print("\n" + "="*50)
#     print("       COMPARISON COMPLETE")
#     print("="*50)



import torch
from collections import OrderedDict

def compare_checkpoints(checkpoint_path1: str, checkpoint_path2: str):
    """
    Compares two PyTorch Lightning checkpoint files to detect architectural,
    hyperparameter, and tensor argument differences.

    Args:
        checkpoint_path1 (str): Path to the first checkpoint file (.ckpt).
        checkpoint_path2 (str): Path to the second checkpoint file (.ckpt).
    """
    try:
        # Load the checkpoints
        print(f"Loading checkpoint from: {checkpoint_path1}")
        ckpt1 = torch.load(checkpoint_path1, map_location='cpu')
        print(f"Loading checkpoint from: {checkpoint_path2}")
        ckpt2 = torch.load(checkpoint_path2, map_location='cpu')
    except FileNotFoundError as e:
        print(f"Error: One of the checkpoint files was not found. {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading checkpoints: {e}")
        return

    # --- Step 1: Compare State Dictionaries (Model Architecture) ---
    print("\n" + "="*50)
    print("      COMPARING MODEL STATE DICTIONARIES (KEYS)")
    print("="*50)

    state_dict1 = ckpt1.get('state_dict', OrderedDict())
    state_dict2 = ckpt2.get('state_dict', OrderedDict())

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Find keys that are unique to each checkpoint
    unique_keys_ckpt1 = keys1 - keys2
    unique_keys_ckpt2 = keys2 - keys1

    if unique_keys_ckpt1:
        print("\nKeys unique to Checkpoint 1:")
        for key in unique_keys_ckpt1:
            print(f"  - {key}")
    
    if unique_keys_ckpt2:
        print("\nKeys unique to Checkpoint 2:")
        for key in unique_keys_ckpt2:
            print(f"  - {key}")

    if not unique_keys_ckpt1 and not unique_keys_ckpt2:
        print("\n✅ The state dictionaries have the same set of keys.")
    else:
        print("\n🚨 WARNING: The models have different architectures. This is the likely cause of your saving error.")
        
    # Compare the shapes and arguments of shared keys
    print("\n" + "="*50)
    print("  COMPARING TENSOR SHAPES & ARGUMENTS FOR COMMON KEYS")
    print("="*50)
    
    common_keys = keys1.intersection(keys2)
    shape_mismatches = []
    device_mismatches = []
    requires_grad_mismatches = []

    for key in common_keys:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        # Compare shapes
        if tensor1.shape != tensor2.shape:
            shape_mismatches.append((key, tensor1.shape, tensor2.shape))
            
        # Compare device
        if tensor1.device != tensor2.device:
            device_mismatches.append((key, tensor1.device, tensor2.device))
            
        # Compare requires_grad
        if tensor1.requires_grad != tensor2.requires_grad:
            requires_grad_mismatches.append((key, tensor1.requires_grad, tensor2.requires_grad))

    if shape_mismatches:
        print("\n🚨 The following layers have mismatched tensor shapes:")
        for key, shape1, shape2 in shape_mismatches:
            print(f"  - Key: {key}")
            print(f"    - Checkpoint 1 shape: {shape1}")
            print(f"    - Checkpoint 2 shape: {shape2}")
    else:
        print("\n✅ All shared layers have matching tensor shapes.")
        
    if device_mismatches:
        print("\n🚨 The following layers have mismatched devices:")
        for key, device1, device2 in device_mismatches:
            print(f"  - Key: {key}")
            print(f"    - Checkpoint 1 device: {device1}")
            print(f"    - Checkpoint 2 device: {device2}")
    else:
        print("\n✅ All shared layers have matching devices.")
        
    if requires_grad_mismatches:
        print("\n🚨 The following layers have mismatched 'requires_grad' values:")
        for key, rg1, rg2 in requires_grad_mismatches:
            print(f"  - Key: {key}")
            print(f"    - Checkpoint 1 requires_grad: {rg1}")
            print(f"    - Checkpoint 2 requires_grad: {rg2}")
    else:
        print("\n✅ All shared layers have matching 'requires_grad' values.")

    # --- Step 2: Compare Hyperparameters ---
    print("\n" + "="*50)
    print("      COMPARING HYPERPARAMETERS")
    print("="*50)

    hparams1 = ckpt1.get('hyper_parameters', {})
    hparams2 = ckpt2.get('hyper_parameters', {})

    hparams_keys1 = set(hparams1.keys())
    hparams_keys2 = set(hparams2.keys())

    unique_hparams1 = hparams_keys1 - hparams_keys2
    unique_hparams2 = hparams_keys2 - hparams_keys1
    
    if unique_hparams1:
        print("\nHyperparameters unique to Checkpoint 1:")
        for key in unique_hparams1:
            print(f"  - {key}: {hparams1.get(key)}")
            
    if unique_hparams2:
        print("\nHyperparameters unique to Checkpoint 2:")
        for key in unique_hparams2:
            print(f"  - {key}: {hparams2.get(key)}")
            
    common_hparams = hparams_keys1.intersection(hparams_keys2)
    
    value_mismatches = []
    for key in common_hparams:
        if hparams1.get(key) != hparams2.get(key):
            value_mismatches.append((key, hparams1.get(key), hparams2.get(key)))
            
    if value_mismatches:
        print("\n🚨 The following common hyperparameters have mismatched values:")
        for key, val1, val2 in value_mismatches:
            print(f"  - Key: {key}")
            print(f"    - Checkpoint 1 value: {val1}")
            print(f"    - Checkpoint 2 value: {val2}")
    
    if not unique_hparams1 and not unique_hparams2 and not value_mismatches:
        print("\n✅ All hyperparameters are identical.")
    else:
        print("\n🚨 WARNING: Hyperparameters are different between the checkpoints.")


    # --- Step 3: Compare Trainer State ---
    print("\n" + "="*50)
    print("      COMPARING TRAINER STATE")
    print("="*50)
    
    trainer_state_keys1 = set(ckpt1.get('trainer_states', {}).keys())
    trainer_state_keys2 = set(ckpt2.get('trainer_states', {}).keys())
    
    all_trainer_state_keys = trainer_state_keys1.union(trainer_state_keys2)

    trainer_state_mismatches = []
    for key in all_trainer_state_keys:
        val1 = ckpt1.get('trainer_states', {}).get(key)
        val2 = ckpt2.get('trainer_states', {}).get(key)
        if val1 != val2:
            trainer_state_mismatches.append((key, val1, val2))
    
    if trainer_state_mismatches:
        print("\n🚨 The following trainer state keys have mismatched values:")
        for key, val1, val2 in trainer_state_mismatches:
            print(f"  - Key: {key}")
            print(f"    - Checkpoint 1 value: {val1}")
            print(f"    - Checkpoint 2 value: {val2}")
    else:
        print("\n✅ All trainer state keys are identical.")


    # --- Step 4: Compare Version Information ---
    print("\n" + "="*50)
    print("      COMPARING VERSIONS")
    print("="*50)

    # Note: Some versions might not have all keys, so we check for existence.
    pl_version1 = ckpt1.get('pytorch-lightning_version', 'N/A')
    pl_version2 = ckpt2.get('pytorch-lightning_version', 'N/A')

    if pl_version1 != pl_version2:
        print(f"\n🚨 PyTorch Lightning versions do not match:")
        print(f"  - Checkpoint 1 version: {pl_version1}")
        print(f"  - Checkpoint 2 version: {pl_version2}")
    else:
        print(f"\n✅ PyTorch Lightning versions match: {pl_version1}")

    torch_version1 = ckpt1.get('torch_version', 'N/A')
    torch_version2 = ckpt2.get('torch_version', 'N/A')

    if torch_version1 != torch_version2:
        print(f"\n🚨 Torch versions do not match:")
        print(f"  - Checkpoint 1 version: {torch_version1}")
        print(f"  - Checkpoint 2 version: {torch_version2}")
    else:
        print(f"\n✅ Torch versions match: {torch_version1}")


    print("\n" + "="*50)
    print("       COMPARISON COMPLETE")
    print("="*50)


# Example Usage:
# Replace 'path/to/old_checkpoint.ckpt' and 'path/to/new_checkpoint.ckpt'
# with the actual paths to your files.
# if __name__ == "__main__":
#     # compare_checkpoints(CHECKPOINTS_DIR + 'splatt3r_3stage_base.ckpt', CHECKPOINTS_DIR[:-5] + 'splatt3r_use_trained.ckpt')
#     compare_checkpoints(CHECKPOINTS_DIR + 'splatt3r_3stage_base.ckpt', CHECKPOINTS_DIR + '25-08-29-02-41-17_epoch=10_step=22473_pred2.ckpt')
