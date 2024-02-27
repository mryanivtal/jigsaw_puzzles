import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    n = 224
    expected_random_distance = 0.5214 * n

    sizes = []
    hardness = []

    for puzzle_part_size in tqdm([224, 112, 56, 32, 16, 8, 4, 1]):
        puzzle_part_size = 1 if puzzle_part_size == 0 else puzzle_part_size

        sizes.append(puzzle_part_size)

        point_avg_diffs = []

        for x in range(n):
            for y in range(n):
                # -- build dist matrix
                dist_matrix = np.zeros([n, n])
                for i in range(n):
                    dist_matrix[i, :] = dist_matrix[i, :] + (i - y) ** 2

                for j in range(n):
                    dist_matrix[:, j] = dist_matrix[:, j] + (j - x) ** 2

                dist_matrix = np.sqrt(dist_matrix)

                # --- Take puzzle rsolution into account
                distance_diff = dist_matrix - expected_random_distance
                part_y_idx = y // puzzle_part_size
                part_x_idx = x // puzzle_part_size
                distance_diff[part_y_idx * puzzle_part_size: (part_y_idx+1) * puzzle_part_size, part_x_idx * puzzle_part_size: (part_x_idx+1) * puzzle_part_size] = 0

                # -- store vals
                avg_diff = abs(distance_diff).sum() / (n**2 - 1)
                point_avg_diffs.append(avg_diff)
                del dist_matrix
                del distance_diff

        point_avg_diffs = np.array(point_avg_diffs).mean()
        hardness.append(point_avg_diffs)

    print()
    sized_hardness = [hardness[i] / sizes[i] for i in range(len(hardness))]
    norm_sized_hardness = [sized_hardness[i] / sized_hardness[-1] for i in range(len(hardness))]

    sqrt_sized_hardness = [hardness[i] / np.sqrt(sizes[i]) for i in range(len(hardness))]
    norm_sqrt_sized_hardness = [sqrt_sized_hardness[i] / sqrt_sized_hardness[-1] for i in range(len(hardness))]


    df = pd.DataFrame()
    df['part_size_pixels'] = sizes
    df['sized_hardness'] = sized_hardness
    df['avg_distance'] = hardness
    df['sqrt_sized_hardness'] = sqrt_sized_hardness
    df['norm_sized_hardness'] = norm_sized_hardness
    df['norm_sqrt_sized_hardness'] = norm_sqrt_sized_hardness

    df.to_csv(r'D:\docs\Study\DSML_IDC\Final project\project\final_paper\analysis_notebook\data\perm_hardness_table.csv')
