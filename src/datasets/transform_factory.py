from torchvision.transforms import transforms


def get_train_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        transforms.RandomErasing(),
    ])
    return transform


def get_infer_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    return transform

