from torch.nn import BCELoss


def get_criterion(params: dict):
    if params['name'] == 'bce_loss':
        criterion = BCELoss()
    else:
        raise NotImplementedError(f'Criterion {params["name"]} is not implemented')

    return criterion
