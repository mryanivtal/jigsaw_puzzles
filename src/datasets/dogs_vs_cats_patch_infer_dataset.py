import json

import torch

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.dogs_vs_cats_patch_train_dataset import DogsVsCatsPatchDataset
from src.jigsaw.jigsaw_scrambler import JigsawScrambler


class DogsVsCatsPatchInferDataset(DogsVsCatsDataset):
    def __init__(self, images_path: str, parts_y: int, parts_x: int, transform=None, transform_for_display=None, shuffle=False):
        super(DogsVsCatsPatchDataset, self).__init__(images_path, transform, transform_for_display=transform_for_display, shuffle=shuffle)

        self.parts_y = int(parts_y)
        self.parts_x = int(parts_x)
        _, self.image_size_y, self.image_size_x = super(DogsVsCatsPatchDataset, self).get_item(0)[0].shape

    def get_item(self, item, for_display: bool=False) -> tuple[torch.Tensor, dict]:
        image, sample_metadata = super(DogsVsCatsPatchDataset, self).get_item(item, for_display=for_display)

        item_patches = JigsawScrambler.get_jigsaw_places_and_patches(image, self.parts_y, self.parts_x)



        return sample_data, sample_metadata


    def __getitem__(self, item):
        image, sample_metadata = self.get_item(item)
        sample_metadata['image_metadata'] = json.dumps(sample_metadata['image_metadata'])

        return image, sample_metadata
