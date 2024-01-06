import random
import unittest
import torch

from src.datasets.jigsaw_puzzle import create_jigsaw_puzzle


class TestJigsawPuzzleCreator(unittest.TestCase):
    def test_jigsaw(self):
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

        jigsaw_tensor = create_jigsaw_puzzle(original_tensor, parts_x, parts_y, shuffled_order)

        reverse_shuffle = {shuffled_order[key]: key for key in shuffled_order.keys()}
        back_tensor = create_jigsaw_puzzle(jigsaw_tensor, parts_x, parts_y, reverse_shuffle)

        self.assertTrue(all(original_tensor.flatten() == back_tensor.flatten()))  # add assertion here


if __name__ == '__main__':
    unittest.main()
