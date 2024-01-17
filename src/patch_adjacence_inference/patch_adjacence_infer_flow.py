import random
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dogs_vs_cats_jigsaw_dataset import DogsVsCatsJigsawDataset
from src.datasets.dogs_vs_cats_patch_infer_dataset import DogsVsCatsPatchInferDataset
from src.datasets.transform_factory import get_predict_transform
from src.jigsaw.jigsaw_scrambler import JigsawScrambler, create_spatial_index_dicts
from src.puzzle_solvers.greedy_solver import GreedySolver
from src.puzzle_solvers.greedy_solver_bgu import GreedySolverBgu
from src.trainer.factories.model_factory import get_model
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
        target_permutation = sample[1]['target']

        pair_patches = sample[0][0]
        pair_relations = sample[0][1]

        # --- Run inference on all pairs in image
        pair_probabilities = l_module(pair_patches).detach().numpy()

        # --- Display some samples of predicted pairs to see how the model is doing
        if False:
            display_patch_pred_samples(pair_patches, pair_probabilities, 5)

        # --- prep data for solver - convert spatial part representation to index
        spatial_to_index, index_to_spatial = create_spatial_index_dicts(parts_y, parts_x)
        pair_relations = [(spatial_to_index[pair[0]], spatial_to_index[pair[1]]) for pair in pair_relations]

        # --- Run solver, get proposed solved permutation
        solved_permutation = GreedySolver(parts_y, parts_x, pair_relations, pair_probabilities).solve()
        solved_permutation = {index_to_spatial[i]: solved_permutation[i] for i in solved_permutation.keys()}

        # --- Display outcomes
        plain_image, _ = super(DogsVsCatsJigsawDataset, dataset).get_item(image_idx, for_display=True)
        scrambled_image, _ = super(DogsVsCatsPatchInferDataset, dataset).get_item(image_idx, for_display=True)

        truth_unscrambled_image = JigsawScrambler._create_jigsaw_tensor_deterministic(scrambled_image, parts_y, parts_x, target_permutation)

        solved_image = JigsawScrambler._create_jigsaw_tensor_deterministic(scrambled_image, parts_y, parts_x, solved_permutation)

        display_image(plain_image)
        # display_image(truth_unscrambled_image)
        display_image(scrambled_image)
        display_image(solved_image)
        print()


def display_patch_pred_samples(pair_patches, pair_probabilities, num_patches):
    import numpy as np
    from src.datasets.dogs_vs_cats_patch_train_dataset import DogsVsCatsPatchTrainDataset
    preds = np.argmax(pair_probabilities, axis=1)
    interesting = np.where(preds != 4)[0]
    pairs_to_show = pair_patches[interesting]

    items_to_show = np.random.randint(0, len(interesting), size=num_patches)
    for i in items_to_show:
        idx = interesting[i]
        pair = [pair_patches[idx, :3, ...], pair_patches[idx, 3:, ...]]
        label = preds[idx]
        image = DogsVsCatsPatchTrainDataset._concatenate_for_display(label, pair)
        display_image(image)



