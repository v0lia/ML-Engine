# import torch
import torch.nn as nn

def get_loss_function(config: dict, device):
    loss_function = config.get("loss_function", "CrossEntropyLoss")
    return getattr(nn, loss_function)().to(device)

# if isinstance(loss_fn, torch.nn.Module):
#     loss_fn = loss_fn.to(device)
# else:
#     print(f"WARNING: loss functiom {loss_fn.__class__.__name__} is not a torch.nn.Module")
