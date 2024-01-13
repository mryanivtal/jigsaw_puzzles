import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset, DogsVsCatsLabels
from src.env_constants import TRAIN_DATA_PATH
from src.datasets.transform_factory import get_train_transform


class MyTestCase(unittest.TestCase):
    def test_train_dataset(self):

        transform = get_train_transform({'resize_y': 224, 'resize_x': 224, 'random_erasing': False})
        train_ds = DogsVsCatsDataset(TRAIN_DATA_PATH, transform=transform)
        self.assertEqual(len(train_ds), 25000)

        num_dogs = len(train_ds.index[train_ds.index['label'] == DogsVsCatsLabels.DOG])
        self.assertEqual(num_dogs, 12500)

        a = train_ds[15]

        train_dl = DataLoader(train_ds, batch_size=50)
        b = train_dl.__iter__().__next__()

        self.assertEqual(list(b[0].shape), [50, 3, 224, 224])
        self.assertEqual(len(b[1]['label']), 50)
        self.assertEqual(len(b[1]['image_metadata']), 50)
        self.assertEqual(len(b[1]['label']), 50)


    def test_labels(self):
        DATA_FOLDER = Path(__file__).parent / Path('resources')

        mixed_ds = DogsVsCatsDataset(DATA_FOLDER)
        self.assertEqual(len(mixed_ds), 9)

        index = mixed_ds.index

        data = [mixed_ds[i] for i in range(len(mixed_ds))]

        print()

if __name__ == '__main__':
    unittest.main()
