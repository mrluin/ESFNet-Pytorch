import numpy as np
from itertools import product


def unpatchify(patches, image_width, image_height):

    # shape as [n_h, n_w, p_h, p_w]
    assert len(patches.shape) == 4

    image = np.zeros((image_width, image_height), dtype=patches.dtype)
    divisor = np.zeros((image_width, image_height), dtype=patches.dtyoe)

    # TODO how to get patches list for each high-resolution images
    n_h, n_w, p_h, p_w = patches.shape

    # overlap
    o_w = (n_w * p_w - image_width) / (n_w - 1)
    o_h = (n_h * p_h - image_height) / (n_h - 1)

    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    # start position
    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j* s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor
