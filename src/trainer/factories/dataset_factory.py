import numpy as np
from torch.utils.data import random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.dogs_vs_cats_jigsaw_dataset import DogsVsCatsJigsawDataset
from src.datasets.dogs_vs_cats_patch_train_dataset import DogsVsCatsPatchDataset
from src.datasets.transform_factory import get_train_transform, get_predict_transform
from src.util_functions.printc import printc


def get_datasets(dataset_params, train_data_path, test_data_path) -> tuple:
    # --- params
    transform_params = dataset_params['transforms']

    # --- fix transform resize params so images can be divided to requested blocks
    if dataset_params['dataset_type'] == 'jigsaw':
        dataset_target = dataset_params['target_type']

        parts_x = dataset_params['scrambler']['parts_x']
        parts_y = dataset_params['scrambler']['parts_y']

        dataset_params['transforms']['resize_x'] = round(dataset_params['transforms']['resize_x'] / parts_x) * parts_x
        dataset_params['transforms']['resize_y'] = round(dataset_params['transforms']['resize_y'] / parts_y) * parts_y

    elif dataset_params['dataset_type'] == 'patch_adjacence':
        parts_x = dataset_params['parts_x']
        parts_y = dataset_params['parts_y']

        dataset_params['transforms']['resize_x'] = round(dataset_params['transforms']['resize_x'] / parts_x) * parts_x
        dataset_params['transforms']['resize_y'] = round(dataset_params['transforms']['resize_y'] / parts_y) * parts_y

        patch_size_x = dataset_params['transforms']['resize_x'] / parts_x
        patch_size_y = dataset_params['transforms']['resize_y'] / parts_y

    train_transform = get_train_transform(transform_params)
    train_transform_for_display = get_train_transform(transform_params, normalize=False)

    predict_transform = get_predict_transform(transform_params)
    predict_transform_for_display = get_predict_transform(transform_params, normalize=False)

    # --- Datasets
    if dataset_params['dataset_type'] == 'plain_images':
        train_val_dataset = DogsVsCatsDataset(train_data_path, transform=train_transform, transform_for_display=train_transform_for_display, shuffle=True)
        test_dataset = DogsVsCatsDataset(test_data_path, transform=predict_transform, transform_for_display=predict_transform_for_display, shuffle=False)

    elif dataset_params['dataset_type'] == 'jigsaw':
        scrambler_params = dataset_params['scrambler']
        train_val_dataset = DogsVsCatsJigsawDataset(train_data_path, scrambler_params, target=dataset_target, transform=train_transform, transform_for_display=train_transform_for_display, shuffle=True)
        test_dataset = DogsVsCatsJigsawDataset(test_data_path, scrambler_params, target=dataset_target, transform=predict_transform, transform_for_display=predict_transform_for_display, shuffle=False)

    elif dataset_params['dataset_type'] == 'patch_adjacence':
        train_val_dataset = DogsVsCatsPatchDataset(train_data_path, patch_size_x=patch_size_x, patch_size_y=patch_size_y, transform=train_transform, transform_for_display=train_transform_for_display, shuffle=True)
        test_dataset = DogsVsCatsPatchDataset(test_data_path, patch_size_x=patch_size_x, patch_size_y=patch_size_y, transform=predict_transform, transform_for_display=predict_transform_for_display, shuffle=False)

    else:
        raise NotImplementedError(f'dataset of type {dataset_params["dataset_type"]} is not available!')

    # --- Split train and validation
    train_val_split_ratio = dataset_params['train_val_split_ratio']
    train_split_th = int(np.floor(train_val_split_ratio * len(train_val_dataset)))
    train_dataset, valid_dataset = random_split(train_val_dataset, [train_split_th, len(train_val_dataset) - train_split_th])

    # --- Debug option for very short run (few batches in each dataloader)
    if dataset_params['short_debug_run']:
        printc.yellow('------------------------------------')
        printc.yellow('Warning - This is a short debug run!')
        printc.yellow('------------------------------------')
        train_dataset, valid_dataset, _ = random_split(train_val_dataset, [15, 15, len(train_val_dataset) - 30])
        test_dataset, _ = random_split(test_dataset, [15, len(test_dataset) - 15])

    return train_dataset, valid_dataset, test_dataset