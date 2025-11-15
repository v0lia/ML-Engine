# dataset_utils.py

import numpy as np
import torch

from src.utils.logger import get_logger

### FOR FASHION-MNIST ###

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def get_label_img(images):  # -> [N,C,H,W] for TensorBoard
    logger = get_logger()
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    if images.ndim == 3:    # Fashion-MNIST: grayscale [N,H,W] -> [N,C,H,W]
        return images.unsqueeze(1)
     
    elif images.ndim == 4:
        if images.shape[1] not in [1,3]:    # [N,H,W,C] -> [N,C,H,W]
            images = images.permute(0,3,1,2)
            
    # if no return made yet:
    logger.error("get_label_img: Unsuported import format")
    raise ValueError("get_label_img: Unsuported import format")    

def normalize_images(images):   # will be needed beyond torchvision datasets
    return images
    # if images.dtype == torch.uint8:
    #     return images.float() / 255.0
    # elif torch.is_floating_point(images):
    #     return images
    # else:
    #     raise TypeError (f"Unsupported tensor dtype for images: {images.dtype}")