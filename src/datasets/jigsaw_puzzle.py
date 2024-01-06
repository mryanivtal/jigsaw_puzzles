import numpy as np
import torch


def create_jigsaw_puzzle(image: torch.Tensor, parts_x: int, parts_y: int, new_order: dict) -> (torch.Tensor, torch.Tensor):
    image_ch, image_x, image_y = image.shape

    assert image_x % parts_x == 0, f'Only equal and whole part cropping is supported, got size_x={image_x}, parts_x={parts_x}'
    assert image_y % parts_y == 0, f'Only equal and whole part cropping is supported, got size_y={image_y}, parts_y={parts_y}'

    split_lines_x = get_split_lines(image_x, parts_x)
    split_lines_y = get_split_lines(image_y, parts_y)

    new_image = torch.zeros_like(image)

    for x_idx, y_idx in new_order.keys():
        new_x_idx, new_y_idx = new_order[(x_idx, y_idx)]

        x_from = split_lines_x[x_idx]
        x_to = split_lines_x[x_idx + 1]

        y_from = split_lines_y[y_idx]
        y_to = split_lines_y[y_idx + 1]

        new_x_from = split_lines_x[new_x_idx]
        new_x_to = split_lines_x[new_x_idx + 1]

        new_y_from = split_lines_y[new_y_idx]
        new_y_to = split_lines_y[new_y_idx + 1]

        new_image[:, new_x_from : new_x_to, new_y_from : new_y_to] = image[:, x_from : x_to, y_from : y_to]

    return new_image


def find_small_big_whole_parts(length, parts_num):
    part_size_small = int(np.floor(length / parts_num))
    part_size_big = int(np.ceil(length / parts_num))

    if not part_size_small == part_size_small:
        num_big = length - parts_num * part_size_small
        num_small = parts_num * part_size_big - length

    else:
        part_size_small = 0
        num_small = 0
        num_big = length // part_size_big

    return num_big, part_size_big, num_small, part_size_small


def get_split_lines(length: int, part_num: int) -> list:
    num_big, size_big, num_small, size_small = find_small_big_whole_parts(length, part_num)
    split_lines = [0]

    if num_big > 0:
        start = split_lines[-1]
        split_lines += [start + i * size_big for i in range(1, num_big+1)]

    if num_small > 0:
        start = split_lines[-1]
        split_lines += [start + i * size_small for i in range(1, num_small+1)]

    return split_lines







