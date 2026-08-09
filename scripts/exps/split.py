# ------------------------------------------
# For development / testing only: add parent directory to python path so we can load the package without installing it
# DO NOT use this if you have installed figuregen via pip
import sys
import os

import json5

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(CURRENT_DIR, "figure-gen"))  # root directory
# -------------------------------------------

import figuregen
from figuregen.util import image
import simpleimageio as sio
import numpy as np

# settings

VERTICAL = True
DEGREE = 20
LINE_WIDTH_PT = 0.5
LINE_COLOR = (102, 204, 255)

# images


def load_image(path):
    if (not os.path.exists(path)):
        # red error
        print("\033[91mError: Image file not found: {}\033[0m".format(path))
        return None

    img = sio.read(path)[:, :, :3]
    return img


blue = [82, 110, 186]
orange = [186, 98, 82]

# generate test images
img_blue = np.tile([x / 255 for x in blue], (320, 640, 1))
img_orange = np.tile([x / 255 for x in orange], (320, 640, 1))


def split(pairs, names, tint_range, tint_img, savefile):
    pair_num = len(pairs) // 2
    split_image = []
    for i in range(pair_num):
        a = pairs[i * 2]
        b = pairs[i * 2 + 1]
        split_image.append(image.SplitImage([a, b], weights=[1.0, 1.0], degree=DEGREE, vertical=VERTICAL))

    split_grid = figuregen.Grid(num_rows=1, num_cols=pair_num)
    for i in range(pair_num):
        split_grid[0, i].set_image(figuregen.PNG(split_image[i].get_image()))
        split_grid[0, i].draw_lines(split_image[i].get_start_positions(),
                                    split_image[i].get_end_positions(),
                                    linewidth_pt=LINE_WIDTH_PT, color=LINE_COLOR)
    split_grid.set_col_titles(figuregen.BOTTOM, names)
    split_grid.set_col_titles(figuregen.TOP, [""] * pair_num)

    split_grid.layout.padding[figuregen.RIGHT] = 1
    split_grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=2.5, offset=0.5)

    tint_grid = figuregen.Grid(num_rows=1, num_cols=1)
    tint_grid[0, 0].set_image(figuregen.PNG(tint_img))
    tint_grid.set_col_titles(figuregen.BOTTOM, [str(tint_range[0])])
    tint_grid.set_col_titles(figuregen.TOP, [str(tint_range[1])])

    tint_grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=4, offset=0)
    tint_grid.layout.column_titles[figuregen.TOP] = figuregen.TextFieldLayout(fontsize=8, size=4, offset=0)

    figuregen.figure([[split_grid, tint_grid]], width_cm=10, filename=savefile)
    print("Save to {}".format(savefile))


if __name__ == "__main__":
    exp_dir = os.path.join(CURRENT_DIR, "images", "exps", "rrs-factors")
    tint_min = 0
    tint_max = json5.load(open(os.path.join(exp_dir, "jetmax.json")))["jetmax"]

    images_pairs = [
        "EARS-Before.png", "EARS-After.png",
        "NRRS-Before.png", "NRRS-After.png",
    ]

    names = ["EARS", "NRRS"]

    pairs = [load_image(os.path.join(exp_dir, image_name)) for image_name in images_pairs]

    tint = load_image(os.path.join(exp_dir, "..", "tint", "tint-vertical.png"))

    # only use first 1/3 width,
    w = tint.shape[1]
    tint[:, 0:w // 3:, :] = 1
    tint[:, w * 2 // 3:, :] = 1

    split(pairs, names, (tint_min, tint_max), tint, os.path.join(exp_dir, "rrs-factors-normalization.pdf"))
