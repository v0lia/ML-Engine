# seed.py

from typing import Union

import torch

from src.utils.logger import get_logger
from src.utils.defaults import default_seed

def set_seeds(config, device: Union[str,torch.device]="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    logger = get_logger()

    try:
        seed = config["seed"]
    except KeyError:
        logger.warning(f'Not found seed in {config}. Using default: {default_seed}')
        seed = default_seed

    logger.info(f"Seed: {seed}")

    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
