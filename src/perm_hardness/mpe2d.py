import numpy as np


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


