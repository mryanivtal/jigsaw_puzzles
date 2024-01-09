from enum import Enum

import numpy as np
import torch


class ScramblerMode(Enum):
    RANDOM_PER_DATASET = 0
    RANDOM_PER_SAMPLE = 1


class PermutationType(Enum):
    RANDOM = 0      # Totally random
    SYMMETRIC = 1   # switch places between patches
    SHIFTED = 2     # shift to one side by n blocks
    FLIPPED = 3     # Flipped on x or y
    BORDER = 4      # border patches randomized, center stays


class JigsawScrambler:
    def __init__(self, jigsaw_params: dict):
        self.params = jigsaw_params


    # TODO:Yaniv: add generate_order_... methods per policy type, then use them in __Getitem__()
    #  if deterministic - store order in self
    #  if same for all - randomize per policy once and store in self as well
    #  if random_per_Sample: generate order each getitem()
    #  Afterwards: write a scramble_by_policy() method that gets image and returns scrambled image, for use by dataset


    @classmethod
    def _create_jigsaw_tensor_deterministic(cls, image: torch.Tensor, parts_x: int, parts_y: int, new_order: dict) -> torch.Tensor:
        """
        Gets a 3d tensor image, num of parts on x and y, and new order, returns a shuffled tensor image
        :param image:
        :param parts_x:
        :param parts_y:
        :param new_order: dict of part shuffle.  e.g. {(1, 0): (3,5)} will move the block with index (x=1, y=0) to location of block with index (x=3, y=5)
        :return:torch.Tensor: new shuffled (Jigsaw) image
        """
        image_ch, image_x, image_y = image.shape

        assert image_x % parts_x == 0, f'Only equal and whole part cropping is supported, got size_x={image_x}, parts_x={parts_x}'
        assert image_y % parts_y == 0, f'Only equal and whole part cropping is supported, got size_y={image_y}, parts_y={parts_y}'

        split_lines_x = cls._get_split_lines(image_x, parts_x)
        split_lines_y = cls._get_split_lines(image_y, parts_y)

        new_image = torch.zeros_like(image)

        for x_idx, y_idx in new_order.keys():
            new_x_idx, new_y_idx = new_order[(x_idx, y_idx)]

            x_from = split_lines_x[x_idx]
            x_to = split_lines_x[x_idx + 1]

            y_from = split_lines_y[y_idx]
            y_to = split_lines_y[y_idx + 1]

            new_x_from = split_lines_x[new_x_idx]
            new_x_to = split_lines_x[new_x_idx + 1]

            new_y_from = split_lines_y[new_y_idx]
            new_y_to = split_lines_y[new_y_idx + 1]

            new_image[:, new_x_from : new_x_to, new_y_from : new_y_to] = image[:, x_from : x_to, y_from : y_to]

        return new_image

    @classmethod
    def _find_small_big_whole_parts(cls, length: int, parts_num: int) -> tuple[int, int, int, int]:
        """
        get length, and num of parts, returns sizes of big and small part and number of each kind to cover the length fully
        :param length:
        :param parts_num:
        :return:
        """
        part_size_small = int(np.floor(length / parts_num))
        part_size_big = int(np.ceil(length / parts_num))

        if not part_size_small == part_size_small:
            num_big = length - parts_num * part_size_small
            num_small = parts_num * part_size_big - length

        else:
            part_size_small = 0
            num_small = 0
            num_big = length // part_size_big

        return num_big, part_size_big, num_small, part_size_small

    @classmethod
    def _get_split_lines(cls, length: int, part_num: int) -> list[int]:
        """
        get length, and num of parts, returns borderlines between the parts
        :param length:
        :param part_num:
        :return:
        """
        num_big, size_big, num_small, size_small = cls._find_small_big_whole_parts(length, part_num)
        split_lines = [0]

        if num_big > 0:
            start = split_lines[-1]
            split_lines += [start + i * size_big for i in range(1, num_big+1)]

        if num_small > 0:
            start = split_lines[-1]
            split_lines += [start + i * size_small for i in range(1, num_small+1)]

        return split_lines







