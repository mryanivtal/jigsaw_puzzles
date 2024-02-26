import torch
from torchvision import transforms


def display_image(image: torch.Tensor):
    transforms.ToPILImage()(image).show()


def save_image(image: torch.Tensor, file_path: str):
    transforms.ToPILImage()(image).save(file_path)
