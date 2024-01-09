import glob
from enum import Enum
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

from torch.utils.data import Dataset

class DogsVsCatsLabels(int, Enum):
    CAT = 0
    DOG = 1

class DogsVsCatsDataset(Dataset):
    def __init__(self, images_path: str, transform=None, cache_data=False, shuffle=False):
        self.cache_data = cache_data
        self.transform = transform

        images_path = Path(images_path)
        assert images_path.exists(), f'data path not found: {images_path}'

        file_list = glob.glob(str(images_path / Path('*.jpg')))
        labels_and_ids = [Path(file).name.split('.')[:2] for file in file_list]

        labels = [item[0] for item in labels_and_ids]
        labels = [DogsVsCatsLabels.DOG if label == 'dog' else DogsVsCatsLabels.CAT for label in labels]

        ids = [item[1] for item in labels_and_ids]

        self.metadata = pd.DataFrame()
        self.metadata['id'] = ids
        self.metadata['path'] = file_list
        self.metadata['label'] = labels

        if shuffle:
            self.metadata = self.metadata.sample(frac=1)

        self.cache = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item_metadata = self.metadata.iloc[item]

        if item in self.cache:
            image, sample_metadata = self.cache[item]
        else:
            image, sample_metadata = self._load_sample_from_disk(item_metadata)
            if self.cache_data:
                self.cache[item] = (image, sample_metadata)

        return image, sample_metadata

    def _load_sample_from_disk(self, item_metadata):
        image = Image.open(item_metadata['path'])
        label = item_metadata['label']
        # --- image metadata
        id = item_metadata['id']
        file_path = item_metadata['path']
        original_size = image.size
        image_mode = image.mode
        label = torch.Tensor([label])
        if self.transform:
            image = self.transform(image)
        sample_metadata = {
            'id': id,
            'file_path': file_path,
            'original_size': str(original_size),
            'image_mode': image_mode,
            'label': label
        }
        return image, sample_metadata
