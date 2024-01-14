import torch.nn
from torch.nn import BCELoss, CrossEntropyLoss


def get_criterion(params: dict):
    if params['name'] == 'bce_loss':
        criterion = BCELoss()
    elif params['name'] == 'cross_entropy':
        criterion = CrossEntropyLoss()
    else:
        raise NotImplementedError(f'Criterion {params["name"]} is not implemented')

    return criterion
