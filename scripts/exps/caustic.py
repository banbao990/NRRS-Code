import os

from matplotlib.pyplot import grid
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
from typing import List
import json5
import argparse
from copy import deepcopy
import numpy as np
import math

# ------------------------------------------
# For development / testing only: add parent directory to python path so we can load the package without installing it
# DO NOT use this if you have installed figuregen via pip
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(CURRENT_DIR, "figure-gen"))  # root directory
# -------------------------------------------

import figuregen
from figuregen import util
import simpleimageio
import os

from main_exps import ARGS, image_read, load_json, scene_name_to_show_map, CROPS_DICT

# -------------------------- Argument Parser -----------------------
MAX_DEPTH = 6
IMAGE_DIR = "images/exps/d6-caustic"
# ------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
REFERENCE_DIR = os.path.join(PROJECT_DIR, "common", "assets", "scenes-nrrs", "references", "d{}".format(MAX_DEPTH))

EXP_DIR = "{}/{}".format(CURRENT_DIR, IMAGE_DIR)
crop_colors = [
    [255, 110, 0],
    [0, 200, 100]
]


def tonemap(img):
    return util.image.lin_to_srgb(img)


def gen_one_scene(scene_name: str, method_names: List[str], method_names_show: List[str]):

    image_names = ["{}_{}.exr".format(scene_name, method_name) for method_name in method_names]
    methods_images = [image_read("{}/{}".format(EXP_DIR, image_name)) for image_name in image_names]
    reference_image = image_read("{}/{}_200000.exr".format(REFERENCE_DIR, scene_name), True)

    methods_images = [tonemap(m) for m in methods_images]
    reference_image = tonemap(reference_image)

    json_paths = ["{}/{}.json".format(EXP_DIR, image_name) for image_name in image_names]
    errors = load_json(json_paths)[0]

    # insert error into method names
    for i in range(len(method_names_show)):
        method_names_show[i] = "{}\\\\{:.2f}".format(method_names_show[i], errors[i])

    # define cropping positions and marker colors

    crops = CROPS_DICT[scene_name]

    methods_images.insert(0, reference_image)
    method_names_show.insert(0, "Reference\\\\RelMSE")

    # ---------- TOP is reference ----------
    top_grid = figuregen.Grid(num_rows=1, num_cols=1)

    # fill grid with image data
    e = top_grid.get_element(0, 0)
    e.set_image(figuregen.PNG(reference_image))

    # Add markers for all crops
    c_idx = 0
    for c in crops:
        e.set_marker(c.get_marker_pos(), c.get_marker_size(),
                     color=crop_colors[c_idx], linewidth_pt=0.5)
        c_idx += 1

    m = scene_name_to_show_map()

    top_grid.set_col_titles(figuregen.TOP, [m.get(scene_name, scene_name)])

    # Specify paddings (unit: mm)
    top_grid.layout.set_padding(left=0.5, right=0.5, bottom=1.0)
    top_grid.layout.column_titles[figuregen.TOP] = figuregen.TextFieldLayout(size=3, vertical_alignment='center')

    # ---------- BOTTOM is different method ----------
    # one row for each crop, one column for each method
    bottom_rows = len(crops)
    bottom_cols = len(methods_images)
    bottom_grid = figuregen.Grid(num_rows=bottom_rows, num_cols=bottom_cols)

    # fill grid with images
    bottom_grid.set_col_titles(figuregen.BOTTOM, method_names_show)
    bottom_grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(size=6, vertical_alignment='center')

    for row in range(bottom_rows):
        for col in range(bottom_cols):
            method_image = methods_images[col]
            image = crops[row].crop(method_image)
            e = bottom_grid.get_element(row, col).set_image(figuregen.PNG(image))
            e.set_frame(linewidth=0.8, color=crop_colors[row])

    # Specify paddings (unit: mm)
    bottom_grid.layout.set_padding(column=0.5, left=0.25, right=0.25, row=0.5)

    return top_grid, bottom_grid


# ---------- V-STACK of Horizontal Figures (create figure) ----------


if __name__ == "__main__":
    scene_names = ['sponza-caustic-finer3-v2-view2', "cylinder-2"]
    method_names = ["ears", "nrrs"]
    method_names_show = ["EARS", "NRRS"]

    v_grids = [[] for _ in range(len(scene_names))]
    for scene_name in scene_names:
        print("Processing scene: {}".format(scene_name))
        top_grid, bottom_grid = gen_one_scene(scene_name, deepcopy(method_names), deepcopy(method_names_show))
        v_grids[0].append(top_grid)
        v_grids[1].append(bottom_grid)

    filename = "{}/caustic.pdf".format(EXP_DIR, scene_name)
    figuregen.figure(v_grids, width_cm=10, filename=filename)
    print("\nSaved to: {}".format(filename))
