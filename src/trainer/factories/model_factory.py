import torch
import torchvision
from torchvision.models import ViT_B_16_Weights, VisionTransformer

from models.cpvt import PCPVT
from src.models.patch_adj_model import PatchAdjModel
from src.models.vision_transformer_posemb_disabled import vit_b_16_noemb as vit_b_16_without_pos_emb
from src.models.vision_transformer_posemb_disabled import VisionTransformer as VisionTransformerWithoutPosEmb


def get_model(params: dict):
    if params['name'] == 'resnet18':
        model = get_resnet18(params)

    elif params['name'] == 'vit_b16_224':
        model = get_vit_b16_224(params)
    elif params['name'] == 'custom_vision_transformer':
        model = get_vision_transformer(params)

    elif params['name'] == 'combined_spatial_edge':
        model = get_combined_spatial_edge(params)

    elif params['name'] == 'pcpvt':
        model = get_pcpvt(params)

    # --- Vision transformers with their positional embedding disabled. Useless
    # elif params['name'] == 'vit_b16_224_without_pos_emb':
    #     model = get_vit_b16_224_without_pos_emb(params)

    # elif params['name'] == 'custom_vision_transformer_without_pos_emb':
    #     model = get_vision_transformer_without_pos_emb(params)

    else:
        raise NotImplementedError(f'Model {params["name"]} is not implemented')

    return model


def get_pcpvt(params: dict):
    img_size = params['img_size']
    patch_size = params['patch_size']
    out_features = params['out_features']

    model = PCPVT(img_size=img_size, patch_size=patch_size, in_chans=int(3), num_classes=out_features)
    return model


def get_inference_normalizer(params: dict):
    if params['inference_normalizer'] == 'softmax':
        return torch.nn.Softmax(dim=1)
    elif params['inference_normalizer'] == 'sigmoid':
        return torch.nn.Sigmoid()
    elif params['inference_normalizer'] == 'logits':
        return None
    else:
        raise NotImplementedError(f'Inference_normalizer {params["inference_normalizer"]} is not implemented')


def get_combined_spatial_edge(params):
    resnet_params = {
        'input_channels': 6,
        'out_features': 5,
        'pretrained': True,
        'checkpoint_path': None,
    }
    resnet = get_resnet18(resnet_params)

    return PatchAdjModel(resnet)

def get_resnet18(params):
    out_features = params.get('out_features', None)
    pretrained = params.get('pretrained', False)
    checkpoint_path = params.get('checkpoint_path', None)
    input_channels = params.get('input_channels', None)

    model = torchvision.models.resnet18(pretrained=pretrained)

    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)

    if input_channels is not None and input_channels != 3:
        model.conv1 = torch.nn.Conv2d(input_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path), map_location='cpu')

    return model


def get_vision_transformer(params):
    """
    Returns a pre-trained vision transformer, with patch size 16x16
    :param params:
    :return:
    """
    input_channels = params.get('input_channels', None)
    image_size = params.get('image_size', None)
    patch_size = params.get('patch_size', None)
    num_layers = params.get('num_layers', None)
    num_heads = params.get('num_heads', None)
    hidden_dim = params.get('hidden_dim', None)
    mlp_dim = params.get('mlp_dim', None)
    num_classes = params.get('out_features', None)
    checkpoint_path = params.get('checkpoint_path', None)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes)

    if input_channels is not None and input_channels != 3:
        model.conv1 = torch.nn.Conv2d(input_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model

def get_vit_b16_224(params: dict):
    """
    Returns a pre-trained vision transformer, with the following params:
    image_size=224
    input_channels=3,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,

    :param params:
    :return:
    """
    out_features = params.get('out_features', None)
    pretrained = params.get('pretrained', False)
    checkpoint_path = params.get('checkpoint_path', None)

    model = torchvision.models.vit_b_16(pretrained=pretrained)

    if out_features is not None:
        model.heads[0] = torch.nn.Linear(in_features=model.heads[0].in_features, out_features=out_features)
        model.num_classes = out_features

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model

def get_vit_b16_224_without_pos_emb(params: dict):
    """
    Returns a pre-trained vision transformer, with the following params:
    image_size=224
    input_channels=3,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,

    :param params:
    :return:
    """
    out_features = params.get('out_features', None)
    pretrained = params.get('pretrained', False)
    checkpoint_path = params.get('checkpoint_path', None)

    model = vit_b_16_without_pos_emb(pretrained=pretrained)

    if out_features is not None:
        model.heads[0] = torch.nn.Linear(in_features=model.heads[0].in_features, out_features=out_features)
        model.num_classes = out_features

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model

def get_vision_transformer_without_pos_emb(params):
    """
    Returns a pre-trained vision transformer, with patch size 16x16
    :param params:
    :return:
    """
    input_channels = params.get('input_channels', None)
    image_size = params.get('image_size', None)
    patch_size = params.get('patch_size', None)
    num_layers = params.get('num_layers', None)
    num_heads = params.get('num_heads', None)
    hidden_dim = params.get('hidden_dim', None)
    mlp_dim = params.get('mlp_dim', None)
    num_classes = params.get('out_features', None)
    checkpoint_path = params.get('checkpoint_path', None)

    model = VisionTransformerWithoutPosEmb(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes)

    if input_channels is not None and input_channels != 3:
        model.conv1 = torch.nn.Conv2d(input_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model