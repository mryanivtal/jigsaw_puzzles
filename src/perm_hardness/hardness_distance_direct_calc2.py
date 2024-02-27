import ast
import json
from multiprocessing import Process
from pathlib import Path

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




def calc_hardness(num_parts: int):

    # --- calc distances and hardness
    scrambler_params = {
        "mode": "different_for_all_samples",
        "parts_x": 5,
        "parts_y": 5,
        "permutation_type": "random"
    }

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

    l1_pair_dists_w16= []
    l2_pair_dists_w16 = []
    linf_pair_dists_w16 = []

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

    l1_pair_dist_w16 = 0
    l2_pair_dist_w16 = 0
    linf_pair_dist_w16 = 0

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

        # --- Distance changes caused by the permutation
        x_dist = np.abs(temp[:, 0] - temp[:, 2])
        y_dist = np.abs(temp[:, 1] - temp[:, 3])

        # --- Normalizers based on - o(i, j)
        norm1 = np.sqrt((temp[:, 0] + 1)**2 + (temp[:, 1] + 1)**2)
        norm2 = norm1 **2
        norm4 = norm2 **2
        norm8 = norm2 **4

        # --- distances
        l1_dist = np.abs(x_dist) + np.abs(y_dist)
        l2_dist = np.sqrt(x_dist **2 + y_dist **2)
        linf_dist = np.maximum(np.abs(x_dist), np.abs(y_dist))

        # --- Weights for new updates
        prev_weight = count / new_count
        new_weight = 1 - prev_weight

        # --- pair distances between active pixel and all others, with all combinations of norms and distances
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

        l1_pair_dist_w16 = prev_weight * l1_pair_dist_w16 + new_weight * (l1_dist/norm8).mean()
        l2_pair_dist_w16 = prev_weight * l2_pair_dist_w16 + new_weight * (l2_dist/norm8).mean()
        linf_pair_dist_w16 = prev_weight * linf_pair_dist_w16 + new_weight * (linf_dist/norm8).mean()

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

    l1_pair_dists_w16.append(l1_pair_dist_w16)
    l2_pair_dists_w16.append(l2_pair_dist_w16)
    linf_pair_dists_w16.append(linf_pair_dist_w16)

    df = pd.DataFrame()
    df['parts'] = [num_parts]
    df['part_size'] = [int(224 / num_parts)]

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

    df['l1_pair_dists_w16'] = l1_pair_dists_w16
    df['l2_pair_dists_w16'] = l2_pair_dists_w16
    df['linf_pair_dists_w16'] = linf_pair_dists_w16

    filepath = rf'D:\docs\Study\DSML_IDC\Final project\project\final_paper\analysis_notebook\data\perm_hardness_table_new.csv'
    df.to_csv(filepath, mode='a', header=not Path(filepath).exists())





if __name__ == '__main__':


    part_options = [1, 2, 4, 8, 14, 28, 56, 112, 224]
    # part_options = [112]

    processes = []
    for num_parts in part_options:
        p = Process(target=calc_hardness, args=(num_parts,))
        print(f'Process started for {num_parts}')
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
