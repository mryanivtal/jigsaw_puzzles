import json

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.jigsaw.jigsaw_scrambler import JigsawScrambler


class DogsVsCatsJigsawDataset(DogsVsCatsDataset):
    def __init__(self, images_path: str, scrambler_params: dict, target: str, transform=None, transform_for_display=None, shuffle=False, seed_scrambler: bool=False):
        super(DogsVsCatsJigsawDataset, self).__init__(images_path, transform, transform_for_display=transform_for_display, shuffle=shuffle)

        assert target in ['probability', 'reverse_permutation', 'original_image']
        self.target = target
        self.scrambler_params = scrambler_params
        self.scrambler = JigsawScrambler(scrambler_params)
        self.seed_scrambler = seed_scrambler

    def get_item(self, item, for_display: bool=False):
        image, sample_metadata = super(DogsVsCatsJigsawDataset, self).get_item(item, for_display=for_display)

        random_seed = item+1 if self.seed_scrambler else None
        scrambled_image, permutation = self.scrambler(image, random_seed=random_seed)
        sample_metadata = sample_metadata.copy()
        sample_metadata['image_metadata']['permutation'] = str(permutation)

        if self.target == 'original_image':
            sample_metadata['target'] = image
        elif self.target == 'reverse_permutation':
            reverse_permutation = {permutation[key]: key for key in permutation.keys()}
            sample_metadata['target'] = reverse_permutation
        elif self.target == 'probability':
            sample_metadata['target'] = sample_metadata['label']

        return scrambled_image, sample_metadata

    def __getitem__(self, item):
        image, sample_metadata = self.get_item(item)
        sample_metadata['image_metadata'] = json.dumps(sample_metadata['image_metadata'])
        return image, sample_metadata
