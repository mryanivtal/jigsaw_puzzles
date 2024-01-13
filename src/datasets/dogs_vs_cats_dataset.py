import glob
import json
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
    def __init__(self, images_path: str, transform=None, transform_for_display=None, cache_data=False, shuffle=False):
        self.cache_data = cache_data
        self.transform = transform
        self.transform_for_display = transform_for_display if transform_for_display else transform

        self._build_index(images_path, shuffle)
        self.cache = {}

    def _build_index(self, images_path, shuffle):
        """
        Load file list from disk, parse labels
        :param images_path:
        :param shuffle:
        :return:
        """
        self.images_path = Path(images_path)
        assert self.images_path.exists(), f'data path not found: {images_path}'

        file_list = glob.glob(str(self.images_path / Path('*.jpg')))

        # --- Parse id and label
        labels_and_ids = [Path(file).name.split('.')[:-1] for file in file_list]
        ids = [item[-1] for item in labels_and_ids]

        labels = [item[0] if len(item) == 2 else None for item in labels_and_ids]
        labels = [DogsVsCatsLabels.DOG if label == 'dog' else DogsVsCatsLabels.CAT if label == 'cat' else None for label in labels]

        self.index = pd.DataFrame()
        self.index['id'] = ids
        self.index['path'] = file_list
        self.index['label'] = labels
        if shuffle:
            self.index = self.index.sample(frac=1)

    def __len__(self):
        return len(self.index)

    def get_item(self, item, for_display: bool=False):
        if for_display:
            item = self._get_item_for_display(item)
        else:
            item = self._get_item_for_model(item)
        return item

    def _get_item_for_model(self, item):
        item_metadata = self.index.iloc[item]

        if item in self.cache:
            image, sample_metadata = self.cache[item]
        else:
            image, sample_metadata = self._load_sample_from_disk(item_metadata)

            if self.transform:
                image = self.transform(image)

            if self.cache_data:
                self.cache[item] = (image, sample_metadata)

        return image, sample_metadata

    def _get_item_for_display(self, item):
        item_metadata = self.index.iloc[item]
        image, sample_metadata = self._load_sample_from_disk(item_metadata)

        if self.transform_for_display:
            image = self.transform_for_display(image)

        return image, sample_metadata

    def __getitem__(self, item):
        image, sample_metadata = self.get_item(item)
        sample_metadata['image_metadata'] = json.dumps(sample_metadata['image_metadata'])
        return image, sample_metadata

    def _load_sample_from_disk(self, item_metadata):
        image = Image.open(item_metadata['path'])
        label = item_metadata['label']

        # --- image metadata
        id = item_metadata['id']
        file_path = item_metadata['path']
        original_size = image.size
        image_mode = image.mode

        sample_metadata = {
            'image_metadata': {
                'id': id,
                'file_path': file_path,
                'original_size': str(original_size),
                'image_mode': image_mode,}
        }

        if not pd.isna(label):
            sample_metadata['label'] = torch.Tensor([label])

        return image, sample_metadata
