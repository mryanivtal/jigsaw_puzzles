import torch.nn
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam, AdamW, SGD


def get_optimizer(params: dict, model):
    if params['name'] == 'adamW':
        lr = params.get('lr', 1e-3)
        optimizer = AdamW(model.parameters(), lr=lr)

    elif params['name'] == 'SGD':
        lr = params.get('lr', 1e-3)
        momentum = params.get('momentum', 0)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    else:
        raise NotImplementedError(f'optimizer {params["name"]} is not implemented')

    return optimizer
