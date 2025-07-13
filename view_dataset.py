import torch
dataset_path = "dataset/scannetpp/test"

file0 = dataset_path + "/000000.torch"

file0_torch = torch.load(file0, map_location='cpu')

print(file0_torch[0].keys())
print(file0_torch[0]["cameras"].shape)
print(file0_torch[0]["key"])