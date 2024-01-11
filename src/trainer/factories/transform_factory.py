from torchvision.transforms import transforms


def get_train_transform(params: dict):
    resize_0 = params['resize_0']
    resize_1 = params['resize_1']

    transform = transforms.Compose([
        transforms.Resize((resize_0, resize_1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        transforms.RandomErasing(),
    ])
    return transform


def get_predict_transform(params: dict):
    resize_0 = params['resize_0']
    resize_1 = params['resize_1']

    transform = transforms.Compose([
        transforms.Resize((resize_0, resize_1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    return transform

