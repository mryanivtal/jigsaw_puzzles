from enum import Enum

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset


class DogsVsCatsJigsawDataset(DogsVsCatsDataset):
    def __init__(self, images_path: str, scrambler_policy: dict, transform=None, cache_data=False, shuffle=False):
        super(DogsVsCatsJigsawDataset, self).__init__(images_path, transform, cache_data, shuffle)

        self.policy = scrambler_policy
        #TODO:Yaniv:Continue


