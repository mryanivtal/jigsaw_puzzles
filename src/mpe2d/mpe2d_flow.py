from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.trainer.factories.dataset_factory import get_datasets
from src.trainer.factories.sample_saver_factory import get_sample_saver

from src.util_functions.util_functions import create_output_dir, save_dict_to_json


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

    sample = train_dataset[0]
    scrambled_image = sample[0]
    plain_image = sample[1]['target']

    plain_image_gl = plain_image.mean(dim=0)
    scrambled_image_gl = scrambled_image.mean(dim=0)

    # --- Extract to method later here
    grain_size = 4    # Tau for coarse graining stage (Grain size)
    emb_dim = 2
    stride = 1

    added_entropy = mpe2d(scrambled_image_gl, emb_dim, grain_size, stride) - mpe2d(plain_image_gl, emb_dim, grain_size, stride)

    print(added_entropy)


def mpe2d(image, emb_dim=2, grain_size=2, stride=1) -> float:
    """
    Multiscale permutation entropy 2D
    :param emb_dim: embedding dimension
    :param grain_size: grain size for coarse grain phase
    :param image: image to claculate
    :param stride: stride of kernel
    :return: float: entropy
    """
    # --- Coarse graining stage
    image_size_y, image_size_x = image.shape
    # todo: replace stage by con2d on rgb image with stride tau later
    i_max = int(np.floor(image_size_y / grain_size))
    j_max = int(np.floor(image_size_x / grain_size))
    cg_image = np.zeros([i_max, j_max])
    for i in range(i_max):
        for j in range(j_max):
            cg_image[i, j] = (1 / grain_size) * image[i * grain_size: (i + 1) * grain_size,
                                                j * grain_size: (j + 1) * grain_size].sum()

    # --- Apply PerEn2D on coarse grained image:
    degrees = []
    for i in range(i_max // stride - (emb_dim - 1)):
        for j in range(j_max // stride - (emb_dim - 1)):
            degree = cg_image[i: i + emb_dim, j: j + emb_dim].flatten().argsort()
            degrees.append(degree)
    uniques = np.unique(np.array(degrees), axis=0, return_counts=True)
    probs = uniques[1] / uniques[1].sum()
    entropy = -1 / np.log(np.math.factorial(emb_dim ** 2)) * np.sum(probs * np.log(probs))
    return entropy









