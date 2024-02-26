from pathlib import Path
from typing import Union

import numpy as np
import torch

from perm_hardness.mpe2d import mpe2d
from perm_hardness.util_funcs import shuffle_pixels_by_probability, shuffle_pixels_within_neighborhoods
from src.trainer.factories.dataset_factory import get_datasets
from src.trainer.factories.sample_saver_factory import get_sample_saver

from src.util_functions.util_functions import create_output_dir, save_dict_to_json
from util_functions.image_utils import display_image


def execute_mpe2d_flow(run_params: dict, project_path: Union[str, Path], train_data_path: Union[str, Path], test_data_path: Union[str, Path], stop_before_fit = False) -> None:
    # --- Handle run param dicts
    trainer_params = run_params['trainer']
    dataset_params = run_params['dataset']
    sample_saver_params = run_params['sample_saver']

    # --- output dir creation
    outputs_path = create_output_dir(project_path, trainer_params['run_name'], trainer_params['add_timestamp_to_out_dir'])
    save_dict_to_json(run_params, Path(outputs_path) / Path('experiment_params.json'))

    # --- Datasets
    print('Creating Datasets')
    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_params, train_data_path, test_data_path)

    # --- Save samples to folder
    sample_saver = get_sample_saver(sample_saver_params)
    sample_saver.save_samples(outputs_path, train_dataset, valid_dataset, test_dataset)

    sample = test_dataset[3]
    sample_for_display = test_dataset.get_item(3, for_display=True)
    scrambled_image = sample[0]
    plain_image = sample[1]['target']

    # # --- mpe2d
    # plain_image_gl = plain_image.mean(dim=0)
    # scrambled_image_gl = scrambled_image.mean(dim=0)
    #
    # # --- Extract to method later here
    # grain_size = 4    # Tau for coarse graining stage (Grain size)
    # emb_dim = 2
    # stride = 1
    #
    # added_entropy = mpe2d(scrambled_image_gl, emb_dim, grain_size, stride) - mpe2d(plain_image_gl, emb_dim, grain_size, stride)
    # print(added_entropy)

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




