from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.dogs_vs_cats_patch_infer_dataset import DogsVsCatsPatchInferDataset
from src.datasets.transform_factory import get_predict_transform
from src.jigsaw_scrambler.jigsaw_scrambler import create_spatial_index_dicts, JigsawScrambler
from src.puzzle_solvers.greedy_solver import GreedySolver
from src.trainer.factories.model_factory import get_model, get_inference_normalizer
from src.trainer.trainer_modules.lightning_wrapper import LightningWrapper
from src.util_functions.util_functions import create_output_dir, save_dict_to_json
from src.utils.image_utils import display_image


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

    # --- Test Dataset
    scrambler_params = dataset_params['scrambler']
    dataset = DogsVsCatsPatchInferDataset(test_data_path, scrambler_params, transform, transform_for_display, concat_dim=9)

    # --- Model
    model_params = run_params['model']
    inference_normalizer = get_inference_normalizer(model_params)
    model = get_model(model_params)

    patches_model_ckpt = run_params['part_adj_ckpt_path']
    l_module = LightningWrapper.load_from_checkpoint(patches_model_ckpt, model=model, optimizer=None, criterion=None, inference_normalizer=inference_normalizer)
    l_module.eval()

    # --- Loop on scrambled images (received as blocks in tensor)
    correct_parts = 0
    total_parts = 0

    for image_idx in tqdm(range(len(dataset))):
        image_idx += 1
        sample = dataset[image_idx]
        target_permutation = sample[1]['target']

        pair_patches = sample[0][0]
        pair_relations = sample[0][1]

        # --- Run inference on all pairs in image
        splits = 25
        num_items = int(np.ceil(pair_patches.shape[0] / splits))
        ranges = [(i*num_items, (i+1)*num_items) for i in range(splits)]

        pair_probabilities = None
        for (a, b) in ranges:
            batch_pair_probabilities = l_module(pair_patches[a:b, ...]).detach().numpy()
            pair_probabilities = batch_pair_probabilities if pair_probabilities is None else np.concatenate((pair_probabilities, batch_pair_probabilities), axis=0)
            del batch_pair_probabilities

        # pair_probabilities = l_module(pair_patches).detach().numpy()

        # --- prep data for solver - convert spatial part representation to index
        spatial_to_index, index_to_spatial = create_spatial_index_dicts(parts_y, parts_x)
        pair_relations = [(spatial_to_index[pair[0]], spatial_to_index[pair[1]]) for pair in pair_relations]

        # --- Run solver, get proposed solved permutation
        solved_permutation, solver_steps = GreedySolver(parts_y, parts_x, pair_relations, pair_probabilities, use_shifter=False, max_iterations=10, stop_at_cluster_size=170).solve()
        solved_permutation = {index_to_spatial[i]: solved_permutation[i] for i in solved_permutation.keys()}

        # --- Display outcomes
        # plain_image, _ = super(DogsVsCatsJigsawDataset, dataset).get_item(image_idx, for_display=True)
        # display_image(plain_image)

        scrambled_image, _ = super(DogsVsCatsPatchInferDataset, dataset).get_item(image_idx, for_display=True)
        display_image(scrambled_image)

        solved_image = JigsawScrambler.create_jigsaw_tensor_deterministic(scrambled_image, parts_y, parts_x, solved_permutation)
        display_image(solved_image)

        # --- display solver iterations and clusters
        channels, image_y, image_x = solved_image.shape
        part_size_y = int(image_y / scrambler_params['parts_y'])
        part_size_x = int(image_x / scrambler_params['parts_x'])
        display_solver_steps(index_to_spatial, part_size_x, part_size_y, parts_x, parts_y, scrambled_image, solver_steps)

        print()


def display_solver_steps(index_to_spatial, part_size_x, part_size_y, parts_x, parts_y, scrambled_image, solver_steps):
    for i, step in enumerate(solver_steps):
        solved_permutation = step['reverse_permutation']
        solved_permutation = {index_to_spatial[i]: solved_permutation[i] for i in solved_permutation.keys()}
        solved_image = JigsawScrambler.create_jigsaw_tensor_deterministic(scrambled_image, parts_y, parts_x,
                                                                          solved_permutation)
        cluster_board = step['clusters_board']
        step_main_cluster = np.bincount(cluster_board.astype(int).flatten()).argmax()

        # --- cluster mask
        channels, image_y, image_x = solved_image.shape

        cluster_mask = torch.ones([image_y, image_x])

        for y_idx in range(parts_y):
            for x_idx in range(parts_x):
                y_start = y_idx * part_size_y
                y_end = (y_idx + 1) * part_size_y
                x_start = x_idx * part_size_x
                x_end = (x_idx + 1) * part_size_x

                cluster_mask[y_start: y_end, x_start: x_end] = 0 if cluster_board[
                                                                        y_idx, x_idx] == step_main_cluster else 1

        cluster_mask = cluster_mask * 0.25
        solved_image_clustered = solved_image.clone()
        solved_image_clustered[0] = torch.minimum(solved_image_clustered[0] + cluster_mask,
                                                  torch.ones_like(solved_image_clustered[0]))
        display_image(solved_image_clustered)


def display_patch_pred_samples(pair_patches, pair_probabilities, num_patches):
    import numpy as np
    from src.datasets.dogs_vs_cats_patch_train_dataset import DogsVsCatsPatchTrainDataset
    preds = np.argmax(pair_probabilities, axis=1)
    interesting = np.where(preds != 4)[0]

    items_to_show = np.random.randint(0, len(interesting), size=num_patches)
    for i in items_to_show:
        idx = interesting[i]
        pair = [pair_patches[idx, :3, ...], pair_patches[idx, 3:, ...]]
        label = preds[idx]
        image = DogsVsCatsPatchTrainDataset._concatenate_for_display(label, pair)
        display_image(image)



