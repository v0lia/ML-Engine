# tensorboard_utils.py

from pathlib import Path

#import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.logger import get_logger

def get_writer(run_dir: Path):
    logger = get_logger()
    tensorboard_dir = (run_dir / "tensorboard").resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tensorboard_dir))
    logger.info(f"[TENSORBOARD] Run results will be saved to: {tensorboard_dir}")
    return writer

def close_writer(writer):
    writer.close()
    
    return

