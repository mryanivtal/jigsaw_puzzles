import ast
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import env_constants
from datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from datasets.transform_factory import get_predict_transform, get_train_transform
from jigsaw_scrambler.jigsaw_scrambler import JigsawScrambler
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
            "parts_x": 5,
            "parts_y": 5,
            "permutation_type": "random"
        }
    }

    # --- Datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_params, train_data_path, test_data_path)
    scrambler = test_dataset.scrambler

    # --- Run
    sample = test_dataset.get_item(3, for_display=True)
    scrambled_image = sample[0]
    plain_image = sample[1]['target']
    perm = ast.literal_eval(sample[1]['image_metadata']['permutation'])




    # --- calc distances and hardness
    scrambler_params = {
        "mode": "different_for_all_samples",
        "parts_x": 5,
        "parts_y": 5,
        "permutation_type": "random"
    }

    l1_dist_sum = []
    l1_dist_std = []
    l2_dist_sum = []
    l2_dist_std = []
    linf_dist_sum = []
    linf_dist_std = []

    l1_pair_dists = []
    l2_pair_dists = []
    linf_pair_dists = []

    l1_pair_dists_w = []
    l2_pair_dists_w = []
    linf_pair_dists_w = []

    l1_pair_dists_w2 = []
    l2_pair_dists_w2 = []
    linf_pair_dists_w2 = []

    l1_pair_dists_w4 = []
    l2_pair_dists_w4 = []
    linf_pair_dists_w4 = []

    l1_pair_dists_w8 = []
    l2_pair_dists_w8 = []
    linf_pair_dists_w8 = []

    part_options = [1, 2, 4, 7, 14, 28, 56, 112, 224]
    # part_options = [112]

    for num_parts in part_options:
        scrambler_params['parts_x'] = num_parts
        scrambler_params['parts_y'] = num_parts
        scrambler = JigsawScrambler(scrambler_params)

        ch, y, x = 1, 224, 224
        plain_pixles = torch.arange(y * x).reshape(1, y, x)
        perm_pixels, perm = scrambler(plain_pixles)

        # --- pixel location movement index
        plain_pix = plain_pixles[0]
        perm_pix = perm_pixels[0]

        pixel_locs = np.zeros([y * x, 4])

        for i in range(y):
            for j in range(x):
                pixel_locs[plain_pix[i, j], 0] = i
                pixel_locs[plain_pix[i, j], 1] = j
                pixel_locs[perm_pix[i, j], 2] = i
                pixel_locs[perm_pix[i, j], 3] = j

        # --- Pixel pairwise distances
        l1_pair_dist = 0
        l2_pair_dist = 0
        linf_pair_dist = 0

        l1_pair_dist_w = 0
        l2_pair_dist_w = 0
        linf_pair_dist_w = 0

        l1_pair_dist_w2 = 0
        l2_pair_dist_w2 = 0
        linf_pair_dist_w2 = 0

        l1_pair_dist_w4 = 0
        l2_pair_dist_w4 = 0
        linf_pair_dist_w4 = 0

        l1_pair_dist_w8 = 0
        l2_pair_dist_w8 = 0
        linf_pair_dist_w8 = 0

        count = 0

        for pix in tqdm(range(pixel_locs.shape[0] - 1)):
            # -- plain image distances
            temp = np.zeros([y * x, 4])

            temp[:, 0] = pixel_locs[:, 0] - pixel_locs[pix, 0]    # original diff between X of all pixels and this specific pix
            temp[:, 1] = pixel_locs[:, 1] - pixel_locs[pix, 1]    # original diff between Y of all pixels and this specific pix
            temp[:, 2] = pixel_locs[:, 2] - pixel_locs[pix, 2]    # perm diff between X of all pixels and this specific pix
            temp[:, 3] = pixel_locs[:, 3] - pixel_locs[pix, 3]    # perm diff between Y of all pixels and this specific pix

            temp = temp[pix+1:, :]  # remove distances already calculated and self distance
            new_count = count + temp.shape[0]

            x_dist = np.abs(temp[:, 0] - temp[:, 2])
            y_dist = np.abs(temp[:, 1] - temp[:, 3])

            norm1 = np.sqrt((temp[:, 0] + 1)**2 + (temp[:, 1] + 1)**2)
            norm2 = norm1 **2
            norm4 = norm2 **2
            norm8 = norm2 **4

            l1_dist = np.abs(x_dist) + np.abs(y_dist)
            l2_dist = np.sqrt(x_dist **2 + y_dist **2)
            linf_dist = np.maximum(np.abs(x_dist), np.abs(y_dist))

            prev_weight = count / new_count
            new_weight = 1 - prev_weight

            l1_pair_dist = prev_weight * l1_pair_dist + new_weight * l1_dist.mean()
            l2_pair_dist = prev_weight * l2_pair_dist + new_weight * l2_dist.mean()
            linf_pair_dist = prev_weight * linf_pair_dist + new_weight * linf_dist.mean()

            l1_pair_dist_w = prev_weight * l1_pair_dist_w + new_weight * (l1_dist/norm1).mean()
            l2_pair_dist_w = prev_weight * l2_pair_dist_w + new_weight * (l2_dist/norm1).mean()
            linf_pair_dist_w = prev_weight * linf_pair_dist_w + new_weight * (linf_dist/norm1).mean()

            l1_pair_dist_w2 = prev_weight * l1_pair_dist_w2 + new_weight * (l1_dist/norm2).mean()
            l2_pair_dist_w2 = prev_weight * l2_pair_dist_w2 + new_weight * (l2_dist/norm2).mean()
            linf_pair_dist_w2 = prev_weight * linf_pair_dist_w2 + new_weight * (linf_dist/norm2).mean()

            l1_pair_dist_w4 = prev_weight * l1_pair_dist_w4 + new_weight * (l1_dist/norm4).mean()
            l2_pair_dist_w4 = prev_weight * l2_pair_dist_w4 + new_weight * (l2_dist/norm4).mean()
            linf_pair_dist_w4 = prev_weight * linf_pair_dist_w4 + new_weight * (linf_dist/norm4).mean()

            l1_pair_dist_w8 = prev_weight * l1_pair_dist_w8 + new_weight * (l1_dist/norm8).mean()
            l2_pair_dist_w8 = prev_weight * l2_pair_dist_w8 + new_weight * (l2_dist/norm8).mean()
            linf_pair_dist_w8 = prev_weight * linf_pair_dist_w8 + new_weight * (linf_dist/norm8).mean()

            count = new_count

        l1_pair_dists.append(l1_pair_dist)
        l2_pair_dists.append(l2_pair_dist)
        linf_pair_dists.append(linf_pair_dist)

        l1_pair_dists_w.append(l1_pair_dist_w)
        l2_pair_dists_w.append(l2_pair_dist_w)
        linf_pair_dists_w.append(linf_pair_dist_w)

        l1_pair_dists_w2.append(l1_pair_dist_w2)
        l2_pair_dists_w2.append(l2_pair_dist_w2)
        linf_pair_dists_w2.append(linf_pair_dist_w2)

        l1_pair_dists_w4.append(l1_pair_dist_w4)
        l2_pair_dists_w4.append(l2_pair_dist_w4)
        linf_pair_dists_w4.append(linf_pair_dist_w4)

        l1_pair_dists_w8.append(l1_pair_dist_w8)
        l2_pair_dists_w8.append(l2_pair_dist_w8)
        linf_pair_dists_w8.append(linf_pair_dist_w8)

        # # --- Pixel self- shift distances
        # l1_dist_m = np.abs(pixel_locs[:, 0] - pixel_locs[:, 2]) + np.abs(pixel_locs[:, 1] - pixel_locs[:, 3])
        # l1_dist_sum.append(l1_dist_m.sum())
        # l1_dist_std.append(l1_dist_m.std())
        #
        # l2_dist_m = np.sqrt((pixel_locs[:, 0] - pixel_locs[:, 2])**2 + (pixel_locs[:, 1] - pixel_locs[:, 3]) ** 2)
        # l2_dist_sum.append(l2_dist_m.sum())
        # l2_dist_std.append(l2_dist_m.std())
        #
        # linf_dist_m = np.maximum(np.abs(pixel_locs[:, 0] - pixel_locs[:, 2]), np.abs(pixel_locs[:, 1] - pixel_locs[:, 3]))
        # linf_dist_sum.append(linf_dist_m.sum())
        # linf_dist_std.append(linf_dist_m.std())

    print()

    df = pd.DataFrame()
    df['parts'] = part_options
    df['part_size'] = [int(224/p) for p in part_options]

    df['l1_pair_dists'] = l1_pair_dists
    df['l2_pair_dists'] = l2_pair_dists
    df['linf_pair_dists'] = linf_pair_dists

    df['l1_pair_dists_w'] = l1_pair_dists_w
    df['l2_pair_dists_w'] = l2_pair_dists_w
    df['linf_pair_dists_w'] = linf_pair_dists_w

    df['l1_pair_dists_w2'] = l1_pair_dists_w2
    df['l2_pair_dists_w2'] = l2_pair_dists_w2
    df['linf_pair_dists_w2'] = linf_pair_dists_w2

    df['l1_pair_dists_w4'] = l1_pair_dists_w4
    df['l2_pair_dists_w4'] = l2_pair_dists_w4
    df['linf_pair_dists_w4'] = linf_pair_dists_w4

    df['l1_pair_dists_w8'] = l1_pair_dists_w8
    df['l2_pair_dists_w8'] = l2_pair_dists_w8
    df['linf_pair_dists_w8'] = linf_pair_dists_w8

    # df['l1_shift_dist'] = l1_dist_sum
    # df['l2_shift_dist'] = l2_dist_sum
    # df['linf_shift_dist'] = linf_dist_sum

    df.to_csv(r'D:\docs\Study\DSML_IDC\Final project\project\final_paper\analysis_notebook\data\perm_hardness_table_new2.csv')






