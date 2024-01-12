from torchvision.transforms import transforms


def get_train_transform(params: dict, normalize=True):
    resize_0 = params['resize_0']
    resize_1 = params['resize_1']

    transform = [
        transforms.Resize((resize_0, resize_1)),
        transforms.ToTensor(),
    ]

    if params['random_erasing']:
        transform = transform + [transforms.RandomErasing()]

    if normalize:
        transform = transform + [transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(transform)
    return transform


def get_predict_transform(params: dict, normalize=True):
    resize_0 = params['resize_0']
    resize_1 = params['resize_1']

    transform = [
        transforms.Resize((resize_0, resize_1)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform = transform + [transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(transform)
    return transform

