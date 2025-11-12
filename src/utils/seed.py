from typing import Union

import torch

def set_seeds(config, device: Union[str,torch.device]="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    torch.manual_seed(config["seed"])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config["seed"])
