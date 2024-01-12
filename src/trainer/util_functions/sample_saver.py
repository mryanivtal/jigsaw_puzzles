import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms


def save_samples_to_output_dir(outputs_path, num_samples_to_save, train_dataset, test_dataset, valid_dataset):
    samples_path = Path(outputs_path) / Path('samples')

    for dataset_idx, dataset in enumerate([test_dataset, train_dataset, valid_dataset]):
        if type(dataset) == torch.utils.data.Subset:
            dataset = dataset.dataset

        dataset_name = ['train', 'validation', 'test'][dataset_idx]
        save_dataset_samples_to_disk(dataset, dataset_name, num_samples_to_save, samples_path)


def save_dataset_samples_to_disk(dataset, dataset_name, num_samples_to_save, samples_path):
    samples_path.mkdir(exist_ok=True, parents=True)

    sample_idxs = np.random.choice(range(len(dataset)), num_samples_to_save, replace=False)
    samples_metadata = []
    for sample_idx in sample_idxs:
        sample_dict = {}

        perm_image, metadata = dataset.get_item_for_display(sample_idx)

        # --- Copy original image to samples path
        unperm_path = Path(metadata['image_metadata']['file_path'])
        unperm_name = [dataset_name] + unperm_path.name.split('.')[:-1] + [unperm_path.name.split('.')[-1]]
        unperm_name = '.'.join(unperm_name)
        unperm_target_path = samples_path / Path(unperm_name)
        shutil.copy(unperm_path, unperm_target_path)

        # --- Save permutation image to samples path
        perm_image = transforms.ToPILImage()(perm_image)
        perm_name = unperm_name.split('.')[:-1] + ['perm'] + [unperm_path.name.split('.')[-1]]
        perm_name = '.'.join(perm_name)

        perm_target_path = samples_path / Path(perm_name)
        perm_image.save(perm_target_path)

        # --- add row to metadata
        sample_dict['id'] = metadata['image_metadata']['id']
        sample_dict['permutation'] = metadata['image_metadata']['permutation']
        samples_metadata.append(sample_dict)
    samples_metadata = pd.DataFrame.from_records(samples_metadata)
    samples_metadata.to_csv(samples_path / Path(f'{dataset_name}.samples.csv'), index=False)
