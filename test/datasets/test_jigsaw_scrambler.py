import random
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.jigsaw_scrambler.jigsaw_scrambler import JigsawScrambler


class TestJigsawScrambler(unittest.TestCase):
    def test_create_jigsaw_tensor_deterministic(self):
        image_size_x = 6
        image_size_y = 6
        image_channels = 2

        parts_x = 2
        parts_y = 3

        original_tensor = torch.arange(image_size_x * image_size_y * image_channels)
        original_tensor = original_tensor.view(image_channels, image_size_x, image_size_y)

        block_list = [(x, y) for y in range(parts_y) for x in range(parts_x)]
        shuffled_list = [(x, y) for y in range(parts_y) for x in range(parts_x)]
        random.shuffle(shuffled_list)
        shuffled_order = {block_list[i]: shuffled_list[i] for i in range(len(block_list))}

        jigsaw_tensor = JigsawScrambler.create_jigsaw_tensor_deterministic(original_tensor, parts_x, parts_y, shuffled_order)

        reverse_shuffle = {shuffled_order[key]: key for key in shuffled_order.keys()}
        back_tensor = JigsawScrambler.create_jigsaw_tensor_deterministic(jigsaw_tensor, parts_x, parts_y, reverse_shuffle)

        self.assertTrue(all(original_tensor.flatten() == back_tensor.flatten()))  # add assertion here


    def test_predefined_permutation_image(self):
        image_path = Path(__file__).parent / Path('resources/22.jpg')

        permutation = {
            (0, 0): (0, 1),
            (0, 1): (0, 0),
            (0, 2): (0, 2),
            (1, 0): (1, 1),
            (1, 1): (1, 0),
            (1, 2): (1, 2),
        }

        scrambler_params = {
            'mode': 'same_for_all_samples',
            '_mode': 'random_per_sample',

            'parts_x': 3,
            'parts_y': 2,

            'permutation_type': 'predefined',
            'predefined_permutation': permutation
        }

        self._permute_and_show_by_params(image_path, scrambler_params)

    def test_random_permutation_image(self):
        image_path = Path(__file__).parent / Path('resources/22.jpg')

        scrambler_params = {
            'mode': 'same_for_all_samples',
            '_mode': 'random_per_sample',

            'parts_x': 4,
            'parts_y': 3,

            'permutation_type': 'random',
        }

        self._permute_and_show_by_params(image_path, scrambler_params)

    def test_switch_permutation_image(self):
        image_path = Path(__file__).parent / Path('resources/22.jpg')

        scrambler_params = {
            'mode': 'same_for_all_samples',
            '_mode': 'random_per_sample',

            'parts_x': 4,
            'parts_y': 3,

            'permutation_type': 'switch',
        }

        self._permute_and_show_by_params(image_path, scrambler_params)

    def _permute_and_show_by_params(self, image_path, scrambler_params):
        image = Image.open(image_path)
        resize_x = 240 // scrambler_params['parts_x'] * scrambler_params['parts_x']
        resize_y = 120 // scrambler_params['parts_y'] * scrambler_params['parts_y']

        transform = transforms.Compose([transforms.Resize((resize_y, resize_x)), transforms.ToTensor()])
        image_tensor = transform(image)
        scrambler = JigsawScrambler(scrambler_params)
        permuted, permutation = scrambler(image_tensor)
        new_image = transforms.ToPILImage()(permuted)
        image.show()
        new_image.show()

    def test_spatial_vs_index_permutation(self):
        permutation = {
            (0, 0): (0, 1),
            (0, 1): (0, 0),
            (0, 2): (0, 2),
            (1, 0): (1, 1),
            (1, 1): (1, 0),
            (1, 2): (1, 2),
        }

        parts_y = 2
        parts_x = 3

        index_perm = JigsawScrambler.spatial_to_index_permutation(permutation)
        spatial_perm = JigsawScrambler.index_to_spatial_permutation(index_perm, parts_y, parts_x)

        assert permutation == spatial_perm
        assert index_perm == [1, 0, 2, 4, 3, 5]



if __name__ == '__main__':
    unittest.main()
