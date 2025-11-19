# get_data.py

# import sys
# from pathlib import Path

# import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.defaults import PROJECT_ROOT, default_dataset_path
from src.utils.logger import get_logger

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # -> [-1, 1]
    ])
    return transform

def get_dataset(mode_bool: bool):
    logger = get_logger()
    dataset = datasets.FashionMNIST(
        root=default_dataset_path.resolve(),
        train=mode_bool,
        download=True,
        transform=get_transform())
        
    logger.info(f"Dataset: {dataset.__class__.__name__}")
    logger.info(f"Train size: {len(dataset)}" if mode_bool else f"Test size: {len(dataset)}")

    return dataset

def get_dataloader(mode: str, config: dict):
    logger = get_logger()
    batch_size = config.get("batch_size", 64)
    mode_bool = True if mode == "train" else False
    dataloader = DataLoader(get_dataset(mode_bool), batch_size=batch_size, shuffle=mode_bool)
    logger.info(f"Batchs size: {batch_size}. Total batches: {len(dataloader)}")
    return dataloader
