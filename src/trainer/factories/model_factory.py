import torch
import torchvision

def get_model(params: dict):
    if params['name'] == 'resnet18':
        pretrained = params['pretrained']
        model = get_resnet18(pretrained=pretrained, out_features=1)
    else:
        raise NotImplementedError(f'Model {params["name"]} is not implemented')

    return model


def get_resnet18(pretrained=False, out_features=None, path=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path))

    return model