# import torch

### FOR FASHION-MNIST ###

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def get_label_img(images):  # -> [N,C,H,W] for TensorBoard
    if images.ndim == 3:    # Fashion-MNIST: grayscale [N,H,W] -> [N,C,H,W]
        return images.unsqueeze(1)
    elif images.ndim == 4:
        if images.size(1) in [1,3]: # [N,C,H,W]
            return images
        else:                       # [N,H,W,C]
            return images.permute(0,3,1,2)

def normalize_images(images):   # will be needed beyond torchvision datasets
    return images
    # if images.dtype == torch.uint8:
    #     return images.float() / 255.0
    # elif torch.is_floating_point(images):
    #     return images
    # else:
    #     raise TypeError (f"Unsupported tensor dtype for images: {images.dtype}")