# get_loss_function.py

# import torch
import torch.nn as nn
from src.utils.logger import get_logger
from src.utils.defaults import default_loss_function_class

def get_loss_function(config: dict, device):
    logger = get_logger()

    try:
        loss_function_str = config["loss_function"]
        loss_function = getattr(nn, loss_function_str)()
    except (KeyError, TypeError):
        logger.warning(f'Not found "loss_function" in config. Usind default: {default_loss_function_class}')
        loss_function = default_loss_function_class()
    except AttributeError:
        logger.warning(f'Not found loss_function "{loss_function_str}" in torch.nn. Usind default: {default_loss_function_class}')
        loss_function = default_loss_function_class()
        
    logger.info(f"Loss_function: {loss_function}")
    return loss_function.to(device)

# if isinstance(loss_fn, torch.nn.Module):
#     loss_fn = loss_fn.to(device)
# else:
#     print(f"WARNING: loss_functiom {loss_fn.__class__.__name__} is not a torch.nn.Module")
