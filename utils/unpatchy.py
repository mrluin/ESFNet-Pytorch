import numpy as np
from itertools import product


def unpatchify(patches, image_h, image_w):

    # shape as [n_h, n_w, p_h, p_w]
    assert len(patches.shape) == 5

    image = np.zeros((image_h, image_w, 3), dtype=patches.dtype)
    divisor = np.zeros((image_h, image_w, 3), dtype=patches.dtype)
    n_h, n_w, p_h, p_w, _ = patches.shape

    # overlap
    o_w = (n_w * p_w - image_w) / (n_w - 1)
    o_h = (n_h * p_h - image_h) / (n_h - 1)

    #assert int(o_w) == o_w
    #assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    # start position
    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i, j, :, :, :]
        image[(i * s_h):(i * s_h) + p_h, (j* s_w):(j * s_w) + p_w, :] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, :] += 1

    return image / divisor
