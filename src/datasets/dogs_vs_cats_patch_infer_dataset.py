import json

import torch

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.dogs_vs_cats_jigsaw_dataset import DogsVsCatsJigsawDataset
from src.datasets.dogs_vs_cats_patch_train_dataset import DogsVsCatsPatchTrainDataset
from src.jigsaw.jigsaw_scrambler import JigsawScrambler


class DogsVsCatsPatchInferDataset(DogsVsCatsJigsawDataset):
    def __init__(self, images_path: str, scrambler_params: dict,  transform=None, transform_for_display=None, shuffle=False, batch=True):
        super(DogsVsCatsPatchInferDataset, self).__init__(images_path, scrambler_params, target='reverse_permutation', transform=transform, transform_for_display=transform_for_display, shuffle=shuffle, seed_scrambler=True)
        self.batch = batch

    def get_item(self, item, for_display: bool=False) -> tuple[torch.Tensor, dict]:
        image, sample_metadata = super(DogsVsCatsPatchInferDataset, self).get_item(item, for_display=for_display)

        item_patches = self.scrambler.get_jigsaw_places_and_patches(image)

        patch_images = [patch['patch'] for patch in item_patches]
        patch_pairs = [[a, b] for idx, a in enumerate(patch_images) for b in patch_images[idx + 1:]]
        patch_pairs = [torch.concat(pair, dim=0) for pair in patch_pairs]

        if self.batch:
            patch_pairs = torch.stack(patch_pairs)

        patch_locations = [patch['location'] for patch in item_patches]
        location_pairs = [[a, b] for idx, a in enumerate(patch_locations) for b in patch_locations[idx + 1:]]

        return (patch_pairs, location_pairs), sample_metadata

    def __getitem__(self, item):
        patch_pairs, sample_metadata = self.get_item(item)
        sample_metadata['image_metadata'] = json.dumps(sample_metadata['image_metadata'])
        return patch_pairs, sample_metadata
