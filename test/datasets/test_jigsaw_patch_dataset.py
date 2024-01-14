import random
import unittest
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset, DogsVsCatsLabels
from src.datasets.dogs_vs_cats_patch_train_dataset import DogsVsCatsPatchDataset
from src.env_constants import TRAIN_DATA_PATH
from src.datasets.transform_factory import get_train_transform, get_predict_transform


class MyTestCase(unittest.TestCase):
    def test_train_dataset(self):

        transform = get_train_transform({'resize_y': 224, 'resize_x': 224, 'random_erasing': False})
        transform_for_display = get_predict_transform({'resize_y': 224, 'resize_x': 224, 'random_erasing': False}, normalize=False)

        train_ds = DogsVsCatsPatchDataset(TRAIN_DATA_PATH, patch_size_x=100, patch_size_y=100, transform=transform, transform_for_display=transform_for_display)
        self.assertEqual(len(train_ds), 25000)

        for i in range(10):
            a_disp = train_ds.get_item(random.randint(0, len(train_ds)-1), for_display=True)
            # a_disp_img = transforms.ToPILImage()(a_disp[0])
            # a_disp_img.show()
            # print(a_disp[1]['label'])

        train_dl = DataLoader(train_ds, batch_size=50)
        b = train_dl.__iter__().__next__()

        print()


if __name__ == '__main__':
    unittest.main()
