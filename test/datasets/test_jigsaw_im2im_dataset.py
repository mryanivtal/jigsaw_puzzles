import unittest

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsLabels
from src.env_constants import TRAIN_DATA_PATH
from src.datasets.transform_factory import get_train_transform, get_predict_transform
from src.datasets.dogs_vs_cats_jigsaw_im2im_dataset import DogsVsCatsJigsawImg2ImgDataset


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
            'resize_0': 224 // scrambler_params['parts_x'] * scrambler_params['parts_x'],
            'resize_1': 224 // scrambler_params['parts_y'] * scrambler_params['parts_y'],
            'random_erasing': False
        }
        transform = get_train_transform(transform_params)
        disp_transform = get_predict_transform(transform_params, normalize=False)

        train_ds = DogsVsCatsJigsawImg2ImgDataset(TRAIN_DATA_PATH, scrambler_params, transform=transform, transform_for_display=disp_transform, cache_data=True)
        self.assertEqual(len(train_ds), 25000)

        num_dogs = len(train_ds.index[train_ds.index['label'] == DogsVsCatsLabels.DOG])
        self.assertEqual(num_dogs, 12500)

        train_dl = DataLoader(train_ds, batch_size=50)
        b = train_dl.__iter__().__next__()

        self.assertEqual(list(b[0].shape), [50, 3, transform_params['resize_0'], transform_params['resize_1']])
        self.assertEqual(len(b[1]['label']), 50)
        self.assertEqual(len(b[1]['image_metadata']), 50)

        # --- Show images
        a = train_ds.get_item_for_display(15)
        original_image = a[0]
        perm_image = a[1]['target']

        original_image = transforms.ToPILImage()(original_image)
        perm_image = transforms.ToPILImage()(perm_image)

        original_image.show()
        perm_image.show()

if __name__ == '__main__':
    unittest.main()
