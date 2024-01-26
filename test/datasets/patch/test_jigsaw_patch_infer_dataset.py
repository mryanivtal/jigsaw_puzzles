import unittest

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsLabels, DogsVsCatsDataset
from src.datasets.dogs_vs_cats_jigsaw_dataset import DogsVsCatsJigsawDataset
from src.datasets.dogs_vs_cats_patch_infer_dataset import DogsVsCatsPatchInferDataset
from src.env_constants import TRAIN_DATA_PATH
from src.datasets.transform_factory import get_train_transform, get_predict_transform
from src.jigsaw_scrambler.jigsaw_scrambler import JigsawScrambler


class MyTestCase(unittest.TestCase):
    def test_dataset(self):
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
        transform = get_predict_transform(transform_params, normalize=False)

        train_ds = DogsVsCatsPatchInferDataset(TRAIN_DATA_PATH, scrambler_params, transform=transform)

        for i in range(4):
            images_patches, metadata = train_ds[15]
            scrambled_image, _ = super(DogsVsCatsPatchInferDataset, train_ds).get_item(15)
            plain_image, _ = super(DogsVsCatsJigsawDataset, train_ds).get_item(15)
            permutation = metadata['target']

            parts_x = train_ds.scrambler.num_parts_x
            parts_y = train_ds.scrambler.num_parts_y
            unscrambled_image = JigsawScrambler.create_jigsaw_tensor_deterministic(scrambled_image, parts_y, parts_x, permutation)

            # --- Present images on screen
            # plain_image = transforms.ToPILImage()(plain_image)
            # plain_image.show()
            scrambled_image = transforms.ToPILImage()(scrambled_image)
            # scrambled_image.show()
            # unscrambled_image = transforms.ToPILImage()(unscrambled_image)
            # unscrambled_image.show()

            self.assertTrue(all((plain_image == unscrambled_image).tolist()))


if __name__ == '__main__':
    unittest.main()
