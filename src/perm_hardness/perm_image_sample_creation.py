import env_constants
from perm_hardness.functions.mpe2d import mpe2d
from perm_hardness.functions.util_funcs import shuffle_pixels_by_probability, shuffle_pixels_within_neighborhoods
from src.trainer.factories.dataset_factory import get_datasets

from util_functions.image_utils import display_image


if __name__ == '__main__':
    # --- Handle run param dicts
    train_data_path = env_constants.TRAIN_DATA_PATH
    test_data_path = env_constants.TEST_DATA_PATH

    dataset_params = {
        "short_debug_run": False,
        "dataset_type": "jigsaw",
        "target_type": "original_image",
        "train_val_split_ratio": 0.8,
        "transforms": {
            "resize_y": 224,
            "resize_x": 224,
            "random_erasing": False
        },
        "scrambler": {
          "mode": "different_for_all_samples",
          "parts_x": 2,
          "parts_y": 2,
          "permutation_type": "random"
        }
    }

    # --- Datasets
    print('Creating Datasets')
    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_params, train_data_path, test_data_path)

    # --- Run

    sample = test_dataset[3]
    sample_for_display = test_dataset.get_item(3, for_display=True)
    scrambled_image = sample[0]
    plain_image = sample[1]['target']

    # --- pixel shift within neighbourhoods
    plain_image = sample_for_display[1]['target'].detach().clone()
    display_image(plain_image)

    image = plain_image
    block_size = 4
    image = shuffle_pixels_within_neighborhoods(block_size, image)

    display_image(image)

    # --- pixel shift by probability
    plain_image = sample_for_display[1]['target'].detach().clone()
    display_image(plain_image)

    image = plain_image
    prob = 0.3
    image = shuffle_pixels_by_probability(image, prob)

    display_image(image)

    print()




