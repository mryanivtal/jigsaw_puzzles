import json
import random

import torch

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset


class DogsVsCatsPatchTrainDataset(DogsVsCatsDataset):
    def __init__(self, images_path: str, patch_size_x: int, patch_size_y: int, transform=None, transform_for_display=None, shuffle=False, concat_dim=0):
        super(DogsVsCatsPatchTrainDataset, self).__init__(images_path, transform, transform_for_display=transform_for_display, shuffle=shuffle)
        self.patch_size_x = int(patch_size_x)
        self.patch_size_y = int(patch_size_y)
        self._concat_dim = concat_dim
        _, self.image_size_y, self.image_size_x = super(DogsVsCatsPatchTrainDataset, self).get_item(0)[0].shape

    @classmethod
    def _crop_image(cls, image: torch.Tensor, x: int, y: int, size_x: int, size_y: int) -> torch.Tensor:
        assert x + size_x < image.shape[2], f'illegal crop, got x={x}, size_x={size_x}, image shape is {image.shape}'
        assert y + size_y < image.shape[1], f'illegal crop, got y={y}, size_y={size_y}, image shape is {image.shape}'

        patch = image[:, y: y+size_y, x: x+size_x]
        return patch

    def get_item(self, item, for_display: bool=False) -> tuple[torch.Tensor, dict]:
        image, sample_metadata = super(DogsVsCatsPatchTrainDataset, self).get_item(item, for_display=for_display)

        # --- select adjacency status for sample and prepare target (vector[5] - 4 sides + 1 not adjacent)
        # labels:
        # 0 - first above second
        # 1 - first to the left of second
        # 2 - first below second
        # 3 - first to the right of second
        # 4 - Not adjacent

        label = int(random.randint(0, 4))

        x_corner_a, y_corner_a, x_corner_b, y_corner_b = self._get_valid_rand_patch_corners(label)

        patch_a = self._crop_image(image, x_corner_a, y_corner_a, self.patch_size_x, self.patch_size_y)
        patch_b = self._crop_image(image, x_corner_b, y_corner_b, self.patch_size_x, self.patch_size_y)

        patches = [patch_a, patch_b]

        # --- prepare metadata and label
        sample_metadata = sample_metadata.copy()
        sample_metadata['label'] = label

        # -- convert label to oneshot
        label_onehot = torch.zeros(5)
        label_onehot[label] = 1
        sample_metadata['target'] = label_onehot

        # --- Concatenate patches for display
        if not for_display:
            sample_data = torch.concat(patches, dim=self._concat_dim)
        else:
            sample_data = self._concatenate_for_display(label, patches)

        return sample_data, sample_metadata

    @classmethod
    def _concatenate_for_display(cls, label, patches) -> torch.Tensor:
        if label == 0:
            sample_data = torch.concat(patches, dim=1)
        elif label == 2:
            sample_data = torch.concat(list(patches.__reversed__()), dim=1)
        elif label == 1:
            sample_data = torch.concat(patches, dim=2)
        elif label == 3:
            sample_data = torch.concat(list(patches.__reversed__()), dim=2)
        elif label == 4:
            sample_data = torch.concat(patches, dim=1)
        else:
            raise RuntimeError('Got illegal label')
        return sample_data

    def _get_valid_rand_patch_corners(self, label) -> (int, int, int, int):
        # --- select locations of patches based on label
        if label == 0:
            corner_y_min = 0
            corner_y_max = self.image_size_y - 2 * self.patch_size_y
        elif label == 2:
            corner_y_min = self.patch_size_y
            corner_y_max = self.image_size_y - self.patch_size_y
        elif label in [1, 3, 4]:
            corner_y_min = 0
            corner_y_max = self.image_size_y - self.patch_size_y
        else:
            raise ValueError('Invalid label')

        if label == 1:
            corner_x_min = 0
            corner_x_max = self.image_size_x - 2 * self.patch_size_x
        elif label == 3:
            corner_x_min = self.patch_size_x
            corner_x_max = self.image_size_x - self.patch_size_x
        elif label in [0, 2, 4]:
            corner_x_min = 0
            corner_x_max = self.image_size_x - self.patch_size_x
        else:
            raise ValueError('Invalid label')

        x_corner_a = random.randint(corner_x_min, corner_x_max - 1)
        y_corner_a = random.randint(corner_y_min, corner_y_max - 1)

        if label == 0:
            x_corner_b = x_corner_a
            y_corner_b = y_corner_a + self.patch_size_y
        elif label == 2:
            x_corner_b = x_corner_a
            y_corner_b = y_corner_a - self.patch_size_y
        elif label == 1:
            x_corner_b = x_corner_a + self.patch_size_x
            y_corner_b = y_corner_a
        elif label == 3:
            x_corner_b = x_corner_a - self.patch_size_x
            y_corner_b = y_corner_a
        elif label == 4:
            x_corner_b = random.randint(corner_x_min, corner_x_max - 1)
            y_corner_b = random.randint(corner_y_min, corner_y_max - 1)
        else:
            raise ValueError('Invalid label')

        return int(x_corner_a), int(y_corner_a), int(x_corner_b), int(y_corner_b)

    def __getitem__(self, item):
        image, sample_metadata = self.get_item(item)
        sample_metadata['image_metadata'] = json.dumps(sample_metadata['image_metadata'])

        return image, sample_metadata
