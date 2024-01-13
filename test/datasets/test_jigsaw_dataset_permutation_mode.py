import unittest

from torch.utils.data import DataLoader

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsLabels
from src.datasets.dogs_vs_cats_jigsaw_dataset import DogsVsCatsJigsawDataset
from src.env_constants import TRAIN_DATA_PATH
from src.datasets.transform_factory import get_train_transform


class MyTestCase(unittest.TestCase):
    def test_train_dataset(self):
        scrambler_params = {
            'mode': 'same_for_all_samples',
            '_mode': 'random_per_sample',
            'parts_x': 4,
            'parts_y': 3,
            'permutation_type': 'random',
        }

        transform_params = {
            'resize_x': 224 // scrambler_params['parts_x'] * scrambler_params['parts_x'],
            'resize_y': 224 // scrambler_params['parts_y'] * scrambler_params['parts_y'],
            'random_erasing': False
        }
        transform = get_train_transform(transform_params)

        train_ds = DogsVsCatsJigsawDataset(TRAIN_DATA_PATH, scrambler_params, target='reverse_permutation', transform=transform)
        self.assertEqual(len(train_ds), 25000)

        num_dogs = len(train_ds.index[train_ds.index['label'] == DogsVsCatsLabels.DOG])
        self.assertEqual(num_dogs, 12500)

        a = train_ds[15]

        train_dl = DataLoader(train_ds, batch_size=50)
        b = train_dl.__iter__().__next__()

        self.assertEqual(list(b[0].shape), [50, 3, transform_params['resize_y'], transform_params['resize_x']])
        self.assertEqual(len(b[1]['label']), 50)
        self.assertEqual(len(b[1]['image_metadata']), 50)

if __name__ == '__main__':
    unittest.main()
