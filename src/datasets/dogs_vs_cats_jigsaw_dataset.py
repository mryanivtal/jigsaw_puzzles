import json
from enum import Enum

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.jigsaw_scrambler import JigsawScrambler


class DogsVsCatsJigsawDataset(DogsVsCatsDataset):
    def __init__(self, images_path: str, scrambler_params: dict, transform=None, transform_for_display=None, cache_data=False, shuffle=False):
        super(DogsVsCatsJigsawDataset, self).__init__(images_path, transform, transform_for_display=transform_for_display, cache_data=cache_data, shuffle=shuffle)

        self.scrambler_params = scrambler_params
        self.scrambler = JigsawScrambler(scrambler_params)

    def get_item(self, item):
        image, sample_metadata = super(DogsVsCatsJigsawDataset, self).get_item(item)
        image, permutation = self.scrambler(image)
        sample_metadata = sample_metadata.copy()
        sample_metadata['image_metadata']['permutation'] = str(permutation)
        return image, sample_metadata

    def get_item_for_display(self, item):
        image, sample_metadata = super(DogsVsCatsJigsawDataset, self).get_item_for_display(item)
        image, permutation = self.scrambler(image)
        sample_metadata = sample_metadata.copy()
        sample_metadata['image_metadata']['permutation'] = str(permutation)
        return image, sample_metadata
