import torch.nn
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam


def get_optimizer(params: dict, model):
    if params['name'] == 'adam':
        lr = params.get('lr', 1e-3)
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f'optimizer {params["name"]} is not implemented')

    return optimizer
