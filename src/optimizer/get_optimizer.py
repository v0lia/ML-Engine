# get_optimizer.py

# import torch
import torch.optim as optim

from src.utils.logger import get_logger
from src.utils.defaults import default_optimizer_class

def get_optimizer(model, config):
    logger = get_logger()
    
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr=1e-3, betas=(0.9,0.999))
    
    try:
        optimizer_str = config["optimizer"]["type"]
        optimizer_class = getattr(optim, optimizer_str)
    except KeyError:
        logger.warning(f'Not found optimizer type in config. Using dedault: {default_optimizer_class}')
        optimizer_class = default_optimizer_class
    except AttributeError:
        logger.warning(f'Not found optimizer "{optimizer_str}" in torch.optim. Using dedault: {default_optimizer_class}')
        optimizer_class = default_optimizer_class

    try:
        params = config["optimizer"]["params"]
    except KeyError:
        logger.warning(f'Not found optimizer params in config. Using PyTorch defaults.')
        params = {}

    optimizer = optimizer_class(model.parameters(), **params)
    logger.info(f"Optimizer: {optimizer.__class__.__name__} | Params: {optimizer.defaults}")
    return optimizer
    