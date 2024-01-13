import torch
import torchvision


def get_model(params: dict):
    if params['name'] == 'resnet18':
        model = get_resnet18(params)
    else:
        raise NotImplementedError(f'Model {params["name"]} is not implemented')

    return model


def get_resnet18(params):
    out_features = params.get('out_features', None)
    pretrained = params.get('pretrained', False)
    output_type = params.get('output_type', 'plain')
    checkpoint_path = params.get('checkpoint_path', None)

    model = torchvision.models.resnet18(pretrained=pretrained)

    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)

        if output_type == 'sigmoid':
            model.fc = torch.nn.Sequential(model.fc, torch.nn.Sigmoid())
        elif output_type == 'softmax':
            model.fc = torch.nn.Sequential(model.fc, torch.nn.Softmax())
        elif output_type == 'plain':
            pass
        else:
            raise NotImplementedError(f'output type {output_type} not supported')

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model
