# modified from figuregen/util/templates.py
import numpy as np


SUPPORTED_ERROR_METRICS = ["relMSE"]


def relative_mse(img, ref, eps=1e-2) -> float:
    """ Computes the relative mean squared error between two images. """
    img = np.array(img, dtype=np.float32, copy=False)
    ref = np.array(ref, dtype=np.float32, copy=False)
    assert img.shape[0] == ref.shape[0], "Images must have the same height"
    assert img.shape[1] == ref.shape[1], "Images must have the same width"
    return np.mean((img - ref) ** 2 / (ref ** 2 + eps))


# mitsuboshi colors!
METHOD_COLORS = [
    [232, 181, 88],
    [5, 142, 78],
    [94, 163, 188],
    [181, 63, 106],
    [255, 0, 255],
    [255, 0, 0],
    [123, 24, 123],
    [0, 255, 0],
    [0, 0, 255],
]

CROP_COLORS = [
    [255, 110, 0],
    [0, 200, 100],
]

ORDER_COLORS = [
    [255, 153, 153],
    [255, 204, 153],
    [255, 248, 173],
]

# format: "[RGB]{255,153,153}"
ORDER_COLORS_LATEX = ["[RGB]{" + ",".join(map(str, color)) + "}" for color in ORDER_COLORS]
ORDER_COLORS_LATEX.append("{white}")  # add white for the default color
