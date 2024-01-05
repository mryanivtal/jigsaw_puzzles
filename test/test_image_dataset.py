import unittest

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset, DogsVsCatsLabels
from src.datasets.transform_factory import get_train_transform
from src.env_constants import *


class MyTestCase(unittest.TestCase):
    def test_train_dataset(self):

        transform = get_train_transform()
        train_ds = DogsVsCatsDataset(TRAIN_DATA_PATH, transform=transform, cache_data=True)
        self.assertEqual(len(train_ds), 25000)

        num_dogs = len(train_ds.metadata[train_ds.metadata['label'] == DogsVsCatsLabels.DOG])
        self.assertEqual(num_dogs, 12500)

        a = train_ds[15]

        train_dl = DataLoader(train_ds, batch_size=50)
        b = train_dl.__iter__().__next__()

        self.assertEqual(list(b[0].shape), [50, 3, 224, 224])
        self.assertEqual(len(b[1]), 50)


if __name__ == '__main__':
    unittest.main()
