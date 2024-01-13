from torchvision.transforms import transforms


def get_train_transform(params: dict, normalize=True):
    resize_x = params['resize_x']
    resize_y = params['resize_y']

    transform = [
        transforms.Resize((resize_y, resize_x)),
        transforms.ToTensor(),
    ]

    if params['random_erasing']:
        transform = transform + [transforms.RandomErasing()]

    if normalize:
        transform = transform + [transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(transform)
    return transform


def get_predict_transform(params: dict, normalize=True):
    resize_x = params['resize_x']
    resize_y = params['resize_y']

    transform = [
        transforms.Resize((resize_y, resize_x)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform = transform + [transforms.Normalize((0.5), (0.5))]

    transform = transforms.Compose(transform)
    return transform

