import random
from enum import Enum

import numpy as np
import torch


class JigsawScrambler:
    def __init__(self, scrambler_params: dict):
        self.params = scrambler_params

        self.same_for_all = True if scrambler_params['mode'] == 'same_for_all_samples' else False

        self.num_parts_x = scrambler_params['parts_x']
        self.num_parts_y = scrambler_params['parts_y']

        if self.same_for_all:
            self.fixed_permutation = self._generate_order_by_params(scrambler_params)

    def permutate_tensor(self, image: torch.Tensor) -> (torch.Tensor, dict):
        if self.same_for_all:
            permutation = self.fixed_permutation
        else:
            permutation = self._generate_order_by_params(self.params)

        perm_image = self._create_jigsaw_tensor_deterministic(image, self.num_parts_x, self.num_parts_y, permutation)

        return perm_image, permutation

    @classmethod
    def _generate_order_by_params(cls, params: dict) -> dict:
        num_parts_x = params['parts_x']
        num_parts_y = params['parts_y']

        permutation_type = params['permutation_type']

        if permutation_type == 'random':
            permutation = cls._generate_random_permutation(num_parts_x, num_parts_y)

        elif permutation_type == 'switch':
            permutation = cls._generate_switch_permutation(num_parts_x, num_parts_y)

        elif permutation_type == 'predefined':
            permutation = params['predefined_permutation']

        elif permutation_type == 'shift':
            # TODO:Yaniv: implement
            raise NotImplementedError(f'Permutation type {permutation_type} is not supported!')

        elif permutation_type == 'border':
            # TODO:Yaniv: implement
            raise NotImplementedError(f'Permutation type {permutation_type} is not supported!')

        else:
            raise NotImplementedError(f'Permutation type {permutation_type} is not supported!')

        return permutation

    @classmethod
    def _generate_random_permutation(cls, num_tiles_x: int, num_tiles_y: int) -> dict:
        """
        Generate total random permutation
        :param num_tiles_x:
        :param num_tiles_y:
        :return:
        """
        original_places = [(x, y) for x in range(num_tiles_x) for y in range(num_tiles_y)]
        target_places = original_places.copy()
        random.shuffle(target_places)
        permutation = {original_places[i]: target_places[i] for i in range(len(original_places))}

        return permutation

    @classmethod
    def _generate_switch_permutation(cls, num_tiles_x: int, num_tiles_y: int) -> dict:
        """
        Generate random permutation by switching places between pairs of tiles
        :param num_tiles_x:
        :param num_tiles_y:
        :return:
        """
        places = [(x, y) for x in range(num_tiles_x) for y in range(num_tiles_y)]

        permutation = {}
        while len(places) > 1:
            item_a = random.choice(places)
            places.remove(item_a)
            item_b = random.choice(places)
            places.remove(item_b)

            permutation[item_a] = item_b
            permutation[item_b] = item_a

        if len(places) == 1:
            permutation[places[0]] = places[0]

        return permutation

    @classmethod
    def _create_jigsaw_tensor_deterministic(cls, image: torch.Tensor, parts_x: int, parts_y: int,
                                            new_order: dict) -> torch.Tensor:
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

            new_image[:, new_x_from: new_x_to, new_y_from: new_y_to] = image[:, x_from: x_to, y_from: y_to]

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
            split_lines += [start + i * size_big for i in range(1, num_big + 1)]

        if num_small > 0:
            start = split_lines[-1]
            split_lines += [start + i * size_small for i in range(1, num_small + 1)]

        return split_lines

    def __call__(self, image: torch.Tensor) -> (torch.Tensor, dict):
        return self.permutate_tensor(image)