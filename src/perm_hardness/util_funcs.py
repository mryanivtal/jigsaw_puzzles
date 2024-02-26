import torch


def shuffle_pixels_within_neighborhoods(block_size, image):
    ch, image_y, image_x = image.shape
    image = image.detach().clone()
    if block_size > 1:
        for y in range(image_y // block_size):
            for x in range(image_x // block_size):
                block = image[:, y * block_size: (y + 1) * block_size, x * block_size: (x + 1) * block_size]
                block = block.reshape(ch, block_size ** 2)
                block = block[:, torch.randperm(block.shape[1])]
                block = block.reshape(ch, block_size, block_size)
                image[:, y * block_size: (y + 1) * block_size, x * block_size: (x + 1) * block_size] = block

    return image

def shuffle_pixels_by_probability(image, probability):
    ch, image_y, image_x = image.shape
    image = image.detach().clone()
    image = image.reshape(ch, image_x * image_y)
    mask = torch.rand_like(image[0, ...]) < probability
    fro = torch.argwhere(mask).squeeze()
    to = fro[torch.randperm(fro.shape[0])]
    image[:, fro] = image[:, to]
    image = image.reshape(ch, image_y, image_x)

    return image