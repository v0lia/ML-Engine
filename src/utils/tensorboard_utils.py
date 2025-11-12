import time
import sys
from pathlib import Path

#import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(sys.path[0]).resolve()

def get_writer(model_name: str, mode: bool):
    result_path = PROJECT_ROOT / "results" / get_folder_name(model_name, mode)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(result_path))
    print(f"Run results will be saved to: {result_path}")
    return writer

def get_folder_name(model_name: str, mode: bool):
    mode_name = "train" if mode else "eval"
    folder_name = mode_name + "_" + model_name + "_" + time.strftime("%Y-%m-%d-%H-%M-%S")
    return folder_name

def close_writer(writer):
    writer.close()
    return

