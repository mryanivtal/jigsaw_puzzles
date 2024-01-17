import random
from enum import Enum

import numpy as np
import torch


class JigsawScrambler:
    def __init__(self, scrambler_params: dict):
        """
        Permutations are in the form of {(y, x): (new_y, new_x), .....}

        :param scrambler_params:
        """
        self.params = scrambler_params
        self.random_seed = None

        self.same_for_all = True if scrambler_params['mode'] == 'same_for_all_samples' else False

        self.num_parts_y = scrambler_params['parts_y']
        self.num_parts_x = scrambler_params['parts_x']

        if self.same_for_all:
            self.fixed_permutation = self._generate_order_by_params(scrambler_params)

    def random_with_seed(self):
        if self.random_seed:
            return random.Random(self.random_seed)
        else:
            return random

    def permutate_tensor(self, image: torch.Tensor, random_seed: int=None) -> (torch.Tensor, dict):
        self.random_seed = random_seed

        if self.same_for_all:
            permutation = self.fixed_permutation
        else:
            permutation = self._generate_order_by_params(self.params)

        perm_image = self._create_jigsaw_tensor_deterministic(image, self.num_parts_y, self.num_parts_x, permutation)

        return perm_image, permutation

    @classmethod
    def spatial_to_index_permutation(cls, permutation: dict):
        parts_y = max([item[0] for item in permutation]) + 1
        parts_x = max([item[1] for item in permutation]) + 1
        index_perm = [permutation[(y, x)][0] * parts_x + permutation[(y, x)][1] for y in range(parts_y) for x in range(parts_x)]

        return index_perm

    @classmethod
    def index_to_spatial_permutation(cls, index_perm: list[int], parts_y: int, parts_x: int):
        natural_perm_list = [(y, x) for y in range(parts_y) for x in range(parts_x)]
        target_perm_list = [(index // parts_x, index - (index // parts_x) * parts_x) for index in index_perm]

        spatial_perm = {natural_perm_list[i]: target_perm_list[i] for i in range(len(natural_perm_list))}
        return spatial_perm

    def _generate_order_by_params(self, params: dict) -> dict:
        num_parts_y = params['parts_y']
        num_parts_x = params['parts_x']

        permutation_type = params['permutation_type']

        if permutation_type == 'random':
            permutation = self._generate_random_permutation(num_parts_y, num_parts_x)

        elif permutation_type == 'switch':
            permutation = self._generate_switch_permutation(num_parts_y, num_parts_x)

        elif permutation_type == 'predefined':
            permutation = params['predefined_permutation']

        elif permutation_type == 'shift':
            permutation = self._generate_random_shift_permutation(num_parts_y, num_parts_x)
            raise NotImplementedError(f'Permutation type {permutation_type} is not supported!')

        elif permutation_type == 'border':
            # TODO:Yaniv: implement
            raise NotImplementedError(f'Permutation type {permutation_type} is not supported!')

        else:
            raise NotImplementedError(f'Permutation type {permutation_type} is not supported!')

        return permutation

    def _generate_random_permutation(self, num_tiles_x: int, num_tiles_y: int) -> dict:
        """
        Generate total random permutation: dict{(y, x): (new_y, new_x)}
        :param num_tiles_x:
        :param num_tiles_y:
        :return:
        """
        original_places = [(x, y) for x in range(num_tiles_x) for y in range(num_tiles_y)]
        target_places = original_places.copy()
        self.random_with_seed().shuffle(target_places)
        permutation = {original_places[i]: target_places[i] for i in range(len(original_places))}

        return permutation

    def _generate_switch_permutation(self, num_tiles_x: int, num_tiles_y: int) -> dict:
        """
        Generate random permutation by switching places between pairs of tiles
        :param num_tiles_x:
        :param num_tiles_y:
        :return:
        """
        places = [(x, y) for x in range(num_tiles_x) for y in range(num_tiles_y)]

        permutation = {}
        while len(places) > 1:
            item_a = self.random_with_seed().choice(places)
            places.remove(item_a)
            item_b = self.random_with_seed().choice(places)
            places.remove(item_b)

            permutation[item_a] = item_b
            permutation[item_b] = item_a

        if len(places) == 1:
            permutation[places[0]] = places[0]

        return permutation

    def _generate_random_shift_permutation(self, num_tiles_x: int, num_tiles_y: int) -> dict:
        """
        Generates random shift permutation on x and y
        :param num_tiles_x:
        :param num_tiles_y:
        :return:
        """
        shift_x = self.random_with_seed().randint(1, num_tiles_x)
        shift_y = self.random_with_seed().randint(1, num_tiles_y)

        permutation = self._generate_deterministic_shift_permutation(num_tiles_x, num_tiles_y, shift_x, shift_y)

        return permutation

    @classmethod
    def _generate_deterministic_shift_permutation(cls, num_tiles_x: int, num_tiles_y: int, shift_x: int, shift_y: int) -> dict:
        places = [(y, x) for x in range(num_tiles_x) for y in range(num_tiles_y)]
        shifted_places = []
        for i, (y, x) in enumerate(places):
            x += shift_x
            x = x + num_tiles_x if x < 0 else x - num_tiles_x if x >= num_tiles_x else x

            y += shift_y
            y = y + num_tiles_y if y < 0 else y - num_tiles_y if y >= num_tiles_y else y
            shifted_places.append((y, x))
        permutation = {places[i]: shifted_places[i] for i in range(len(shifted_places))}
        return permutation

    @classmethod
    def _create_jigsaw_tensor_deterministic(cls, image: torch.Tensor, parts_y: int, parts_x: int,
                                            new_order: dict) -> torch.Tensor:
        """
        Gets a 3d tensor image, num of parts on x and y, and new order, returns a shuffled tensor image
        :param image:
        :param parts_y:
        :param parts_x:
        :param new_order: dict of part shuffle.  e.g. {(1, 0): (3,5)} will move the block with index (y=1, x=0) to location of block with index (y=3, x=5)
        :return:torch.Tensor: new shuffled (Jigsaw) image
        """
        image_ch, image_x, image_y = image.shape

        assert image_x % parts_y == 0, f'Only equal and whole part cropping is supported, got size_x={image_x}, parts_y={parts_y}'
        assert image_y % parts_x == 0, f'Only equal and whole part cropping is supported, got size_y={image_y}, parts_x={parts_x}'

        split_lines_x = cls._get_split_lines(image_x, parts_y)
        split_lines_y = cls._get_split_lines(image_y, parts_x)

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

    def get_jigsaw_places_and_patches(self, image: torch.Tensor) -> list[dict]:
        return self._get_jigsaw_places_and_patches(image, self.num_parts_y, self.num_parts_x)

    @classmethod
    def _get_jigsaw_places_and_patches(cls, image: torch.Tensor, parts_y: int, parts_x: int) -> list[dict]:
        """
        Returns a dict of locations and patches from the image, split to (parts_y * parts_x) patches
        :param image:
        :param parts_y:
        :param parts_x:
        :return:torch.Tensor: new shuffled (Jigsaw) image
        """
        image_ch, image_x, image_y = image.shape

        assert image_x % parts_y == 0, f'Only equal and whole part cropping is supported, got size_x={image_x}, parts_y={parts_y}'
        assert image_y % parts_x == 0, f'Only equal and whole part cropping is supported, got size_y={image_y}, parts_x={parts_x}'

        split_lines_x = cls._get_split_lines(image_x, parts_y)
        split_lines_y = cls._get_split_lines(image_y, parts_x)

        natural_permutation = [(y, x) for y in range(parts_y) for x in range(parts_x)]

        patch_list = []

        for x_idx, y_idx in natural_permutation:

            x_from = split_lines_x[x_idx]
            x_to = split_lines_x[x_idx + 1]

            y_from = split_lines_y[y_idx]
            y_to = split_lines_y[y_idx + 1]

            patch = image[:, x_from: x_to, y_from: y_to]
            location = (y_idx, x_idx)

            patch_list.append({'location': location, 'patch': patch})

        return patch_list

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

    def __call__(self, image: torch.Tensor, random_seed=None) -> (torch.Tensor, dict):
        return self.permutate_tensor(image, random_seed)


def create_spatial_index_dicts(parts_y:int, parts_x:int) -> (dict, dict):
    index = list(range(parts_y * parts_x))
    spatial = [(y, x) for y in range(parts_y) for x in range(parts_x)]

    spatial_to_index = {spatial[i]: index[i] for i in range(parts_y * parts_x)}
    index_to_spatial = {spatial_to_index[key]: key for key in spatial_to_index.keys()}

    return spatial_to_index, index_to_spatial