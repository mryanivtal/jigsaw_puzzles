import torch
import torchvision


def get_model(params: dict):
    if params['name'] == 'resnet18':
        model = get_resnet18(params)
    else:
        raise NotImplementedError(f'Model {params["name"]} is not implemented')

    return model


def get_inference_normalizer(params: dict):
    if params['inference_normalizer'] == 'softmax':
        return torch.nn.Softmax(dim=1)
    elif params['inference_normalizer'] == 'sigmoid':
        return torch.nn.Sigmoid()


def get_resnet18(params):
    out_features = params.get('out_features', None)
    pretrained = params.get('pretrained', False)
    output_type = params.get('output_type', 'plain')
    checkpoint_path = params.get('checkpoint_path', None)
    input_channels = params.get('input_channels', None)

    model = torchvision.models.resnet18(pretrained=pretrained)

    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)

        if input_channels is not None and input_channels != 3:
            model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if output_type == 'sigmoid':
            model.fc = torch.nn.Sequential(model.fc, torch.nn.Sigmoid())
        elif output_type == 'softmax':
            model.fc = torch.nn.Sequential(model.fc, torch.nn.Softmax(dim=1))
        elif output_type == 'plain':
            pass
        else:
            raise NotImplementedError(f'output type {output_type} not supported')

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model
