from pathlib import Path

import numpy as np
import torch
import lightning as L
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import env_constants
from src.datasets.dogs_vs_cats_patch_infer_dataset import DogsVsCatsPatchInferDataset
from src.datasets.transform_factory import get_predict_transform
from src.trainer.factories.model_factory import get_model
from src.trainer.trainer_modules.lightning_wrapper import LightningWrapper
from src.util_functions.util_functions import create_output_dir, save_dict_to_json


def execute_infer_flow(run_params, project_path, test_data_path):
    # --- Handle run param dicts
    dataset_params = run_params['dataset']
    trainer_params = run_params['inference']

    # --- output dir creation
    outputs_path = create_output_dir(project_path, trainer_params['run_name'], trainer_params['add_timestamp_to_out_dir'])
    save_dict_to_json(run_params, Path(outputs_path) / Path('experiment_params.json'))

    # --- Transforms
    parts_x = dataset_params['scrambler']['parts_x']
    parts_y = dataset_params['scrambler']['parts_y']

    dataset_params['transforms']['resize_x'] = round(dataset_params['transforms']['resize_x'] / parts_x) * parts_x
    dataset_params['transforms']['resize_y'] = round(dataset_params['transforms']['resize_y'] / parts_y) * parts_y

    transform_params = dataset_params['transforms']
    transform = get_predict_transform(transform_params)
    transform_for_display = get_predict_transform(transform_params, normalize=False)

    # --- Dataset
    scrambler_params = dataset_params['scrambler']
    dataset = DogsVsCatsPatchInferDataset(test_data_path, scrambler_params, transform, transform_for_display)

    # --- Dataloader
    persistent_workers = True if trainer_params['num_workers'] > 0 else False
    dataloader = DataLoader(dataset, batch_size=1, num_workers=trainer_params['num_workers'], persistent_workers=persistent_workers)

    # --- Model
    part_adj_model_params = run_params['models']['part_adj_params']
    model = get_model(part_adj_model_params)

    patches_model_ckpt = run_params['models']['part_adj_ckpt_path']
    l_module = LightningWrapper.load_from_checkpoint(patches_model_ckpt, model=model, optimizer=None, criterion=None)
    l_module.eval()


    # --- Loop on scrambled images (received as blocks in tensor)
    for image_idx in tqdm(range(len(dataset))):
        sample = dataset[image_idx]
        image_target_permutation = sample[1]['target']
        pair_patches = sample[0][0]
        pair_relations_scrambled = sample[0][1]

        # --- Run inference on all pairs in image
        pair_probabilities = l_module(pair_patches).detach().numpy()

        # --- Run solver, get proposed solved permutation
        solved_permutation = greedy_solver(parts_y, parts_x, pair_relations_scrambled, pair_probabilities)


def greedy_solver(size_y: int, size_x: int, pair_relations, pair_probabilities) -> dict:
        # --- Greedy solver
        parts_to_place = list({relation[0] for relation in pair_relations}.union({relation[1] for relation in pair_relations}))

        placed_parts = []
        part = parts_to_place.pop(np.random.randint(len(parts_to_place)))
        placed_parts.append(part)

        #TODO: store free sides for each placed part
        # iterate on free sides of place parts:
        #   Choose the empty slot with the top suiting probability
        #   Place there the most suiting part
        #   Update free slots (both sides of the axis!)
        #   If no pore slots (and parts) - move to next stage of solving






def find_pair_probabilities_for_part(part, pair_relations_list: list, pair_probabilities_list: np.ndarray) -> list:
    """
    returns indexes of all relations including the part, flipped if necessary so requested part is first
    :param part: seed part
    :return: list of related parts with relation code, flipped if needed to suit (part, related_part)
    """
    related_part_straight = [p for p in pair_relations_list[0]]
    related_part_flipped = [p for p in pair_relations_list[0]]




    print()

        # TODO:Yaniv: continue from here:
        #  Rebuild image from blocks using probabilities
        #  show on screen original vs. reconstructed, maybe save some samples
        #  next step - run classification model on image and get label
        #  Create stats / plots on success


