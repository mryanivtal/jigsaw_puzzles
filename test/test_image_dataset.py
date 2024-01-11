import unittest

from torch.utils.data import DataLoader

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset, DogsVsCatsLabels
from src.datasets.transform_factory import get_train_transform
from src.runners.env_constants import *


class MyTestCase(unittest.TestCase):
    def test_train_dataset(self):

        transform = get_train_transform()
        train_ds = DogsVsCatsDataset(TRAIN_DATA_PATH, transform=transform, cache_data=True)
        self.assertEqual(len(train_ds), 25000)

        num_dogs = len(train_ds.index[train_ds.index['label'] == DogsVsCatsLabels.DOG])
        self.assertEqual(num_dogs, 12500)

        a = train_ds[15]

        train_dl = DataLoader(train_ds, batch_size=50)
        b = train_dl.__iter__().__next__()

        self.assertEqual(list(b[0].shape), [50, 3, 224, 224])
        self.assertEqual(len(b[1]['label']), 50)
        self.assertEqual(len(b[1]['original_size']), 50)

    def test_labels(self):
        DATA_FOLDER = Path(__file__).parent / Path('resources')

        mixed_ds = DogsVsCatsDataset(DATA_FOLDER, cache_data=True)
        self.assertEqual(len(mixed_ds), 9)

        index = mixed_ds.index

        data = [mixed_ds[i] for i in range(len(mixed_ds))]

        print()

if __name__ == '__main__':
    unittest.main()
