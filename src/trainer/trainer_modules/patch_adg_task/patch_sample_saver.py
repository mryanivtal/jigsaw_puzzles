import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

class PatchSampleSaver:
    def __init__(self, params: dict):
        self.num_samples_to_save = params['num_samples_to_save']

    def save_samples(self, outputs_path, train_dataset, valid_dataset, test_dataset):
        samples_path = Path(outputs_path) / Path('samples')

        for dataset_idx, dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
            if type(dataset) == torch.utils.data.Subset:
                dataset = dataset.dataset

            dataset_name = ['train', 'validation', 'test'][dataset_idx]
            self.save_dataset_samples_to_disk(dataset, dataset_name, self.num_samples_to_save, samples_path)

    @classmethod
    def save_dataset_samples_to_disk(cls, dataset, dataset_name, num_samples_to_save, samples_path):
        samples_path.mkdir(exist_ok=True, parents=True)

        sample_idxs = np.random.choice(range(len(dataset)), num_samples_to_save, replace=False)
        samples_metadata = []
        for sample_idx in sample_idxs:
            sample_dict = {}

            image, metadata = dataset.get_item(sample_idx, for_display=True)

            # --- Copy original image to samples path
            image_path = Path(metadata['image_metadata']['file_path'])
            image_name = [dataset_name] + image_path.name.split('.')[:-1] + [image_path.name.split('.')[-1]]
            image_name = '.'.join(image_name)
            image_target_path = samples_path / Path(image_name)
            image = transforms.ToPILImage()(image)
            image.save(image_target_path)

            # --- add row to metadata
            sample_dict['id'] = metadata['image_metadata']['id']
            sample_dict['label'] = metadata['label']
            samples_metadata.append(sample_dict)

        samples_metadata = pd.DataFrame.from_records(samples_metadata)
        samples_metadata.to_csv(samples_path / Path(f'{dataset_name}.samples.csv'), index=False)


    @staticmethod
    def concatenate_for_display(label, patches) -> torch.Tensor:
        if label == 0:
            sample_data = torch.concat(patches, dim=1)
        elif label == 2:
            sample_data = torch.concat(list(patches.__reversed__()), dim=1)
        elif label == 1:
            sample_data = torch.concat(patches, dim=2)
        elif label == 3:
            sample_data = torch.concat(list(patches.__reversed__()), dim=2)
        elif label == 4:
            sample_data = torch.concat(patches, dim=1)
        else:
            raise RuntimeError('Got illegal label')
        return sample_data