import torch
from torchvision import transforms


def display_image(image: torch.Tensor):
    transforms.ToPILImage()(image).show()
