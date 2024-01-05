import glob
from enum import Enum
from pathlib import Path
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class DogsVsCatsLabels(int, Enum):
    CAT = 0
    DOG = 1

class DogsVsCatsDataset(Dataset):
    def __init__(self, images_path: str, transform=None, cache_data=True):
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

        self.cache = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item_metadata = self.metadata.iloc[item]

        if item in self.cache:
            image = self.cache[item]
        else:
            image = Image.open(item_metadata['path'])

            if self.transform:
                image = self.transform(image)

            if self.cache_data:
                self.cache[item] = image

        label = item_metadata['label']

        return image, label








