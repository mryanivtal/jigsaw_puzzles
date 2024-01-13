import torch
import torchvision

from src.cond_diffusion_trainer.src.models.xunet import XUNet


def get_model(params: dict):
    if params['name'] == 'xnet':
        model = XUNet(H=256, W=256, ch=128)
    else:
        raise NotImplementedError(f'Model {params["name"]} is not implemented')

    return model


