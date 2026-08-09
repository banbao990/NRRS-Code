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
from figuregen.util.image import Cropbox
from crops import MyCropComparison
from utils import METHOD_COLORS

# -------------------------------------------
USE_LATEX = True
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

# for cbox, should be 5
CBOX = False

ARGS = None

CROPS_DICT = {
    "tropical_bedroom_cx": [Cropbox(top=329, left=421, height=96, width=128, scale=5), Cropbox(top=526, left=730, height=96, width=128, scale=5)],
    "noenv_apartment2": [Cropbox(top=516, left=283, height=96, width=128, scale=5), Cropbox(top=50, left=644, height=96, width=128, scale=5)],
    "apartment1": [Cropbox(top=382, left=466, height=96, width=128, scale=5), Cropbox(top=263, left=432, height=96, width=128, scale=5)],
    "apartment3": [Cropbox(top=407, left=533, height=96, width=128, scale=5), Cropbox(top=188, left=580, height=96, width=128, scale=5)],
    "sponza-glossy": [Cropbox(top=590, left=615, height=96, width=128, scale=5), Cropbox(top=277, left=595, height=96, width=128, scale=5)],
    "sponza-view2": [Cropbox(top=410, left=600, height=96, width=128, scale=5), Cropbox(top=217, left=776, height=96, width=128, scale=5)],
    "sponza-caustic": [Cropbox(top=403, left=559, height=96, width=128, scale=5), Cropbox(top=575, left=743, height=96, width=128, scale=5)],
    "sponza-caustic-lowf": [Cropbox(top=403, left=559, height=96, width=128, scale=5), Cropbox(top=575, left=743, height=96, width=128, scale=5)],
    "bistro-interior": [Cropbox(top=270, left=540, height=96, width=128, scale=5), Cropbox(top=495, left=731, height=96, width=128, scale=5)],
    "sun-temple-v2-view2": [Cropbox(top=585, left=411, height=96, width=128, scale=5), Cropbox(top=506, left=845, height=96, width=128, scale=5)],
    "apartment1-v2-view2": [Cropbox(top=369, left=1103, height=96, width=128, scale=5), Cropbox(top=561, left=774, height=96, width=128, scale=5)],
    "m1-f210": [Cropbox(top=535, left=944, height=96, width=128, scale=5), Cropbox(top=376, left=323, height=96, width=128, scale=5)],
    "sun-temple": [Cropbox(top=350, left=604, height=96, width=128, scale=5), Cropbox(top=140, left=490, height=96, width=128, scale=5)],
    "cornell-box2": [Cropbox(top=120, left=120, height=96, width=128, scale=5), Cropbox(top=473, left=330, height=96, width=128, scale=5)],
    "sponza-caustic-finer3-v2-view2": [Cropbox(top=501, left=716, height=96, width=128, scale=5), Cropbox(top=622, left=608, height=96, width=128, scale=5)],
    "cylinder-2": [Cropbox(top=328, left=484, height=96, width=128, scale=5), Cropbox(top=453, left=530, height=96, width=128, scale=5)],
    "san-miguel-view2": [Cropbox(top=323, left=569, height=96, width=128, scale=5), Cropbox(top=224, left=825, height=96, width=128, scale=5)],
}
CROPS_DICT["sun-temple-v2-lighter-view2"] = CROPS_DICT["sun-temple-v2-view2"]
# -------------------------------------------


def image_read(filename, is_reference=False):
    if not os.path.exists(filename):
        if is_reference:
            filename_ori = filename
            dirs = os.path.dirname(filename)
            filename = os.path.basename(filename).split(".")[0].split("_")[:-1]
            # check all files in dirs, begin with filename
            for file in os.listdir(dirs):
                # start with filename, and left is all digits
                prefix = "_".join(filename)
                if file.startswith(prefix) and file.endswith(".exr"):
                    # check if the rest is all digits
                    rest = file[len(prefix) + 1:-4]  # remove .exr
                    if rest.isdigit():
                        filename = os.path.join(dirs, file)
                        break
            print("File not found: {}. Using: {}".format(filename_ori, filename))
        else:
            # raise FileNotFoundError("File not found: {}".format(filename))
            print("File not found: {}".format(filename))
            return []

    img = cv.imread(filename, cv.IMREAD_UNCHANGED)[..., 0:3]
    img = img[:, :, ::-1]
    return img


def error_string(i, errors):
    return f"{errors[i]:.3f} ({errors[i] / errors[0]:.3f}x)"


def load_json(json_paths) -> List[List[float]]:
    ret = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json5.load(f)
            # ['data', 'infer-time', 'rays', 'timepoints', 'timesteps']

            # [#1] relMSE
            relMSE = data["data"][-1]["RelMSE"]

            # [#2] rayEffInv
            rays = data["rays"]
            # infer_time = data["infer-time"]
            # 720p
            ray_count = rays / (1280 * 720)  # * (60 / infer_time)
            rayEffInv = relMSE * ray_count

            ret.append([relMSE * 1000, rayEffInv])

    # transpose the list of lists
    ret = list(map(list, zip(*ret)))
    return ret


##############################################################
################### [Fig1 Main Experiment] ###################
##############################################################


def generate_figure_one_scene(scene_name, only_relMSE=False, header=True):
    method_names_show = deepcopy(ARGS.method_names_show)
    method_names_show.insert(0, "Reference")

    image_names = ["{}_{}.exr".format(scene_name, method_name) for method_name in ARGS.method_names]
    methods_images = [image_read("{}/{}".format(ARGS.exp_dir, image_name)) for image_name in image_names]

    for i in range(len(methods_images)):
        if len(methods_images[i]) == 0:
            # red warning
            print("\033[91m[Warning]: scene `{}` is missing an image.\033[0m".format(image_names[i]))
            return None

    reference_image = image_read("{}/{}_200000.exr".format(ARGS.reference_dir, scene_name), True)

    # jsons
    json_paths = ["{}/{}.json".format(ARGS.exp_dir, image_name) for image_name in image_names]
    errors = load_json(json_paths)
    metrics = ["RelMSE ($\\times10^{-3}$)", "RayEffInv"] if USE_LATEX else ["RelMSE (x1e-3)", "RayEffInv"]

    if CBOX:  # for cbox, error is so small, should be 5
        metrics = ["RelMSE ($\\times10^{-5}$)", "RayEffInv"] if USE_LATEX else ["RelMSE (x1e-5)", "RayEffInv"]
        errors[0] = [e * (1e5 / 1e3) for e in errors[0]]

    if only_relMSE:
        metrics = metrics[0:1]
        errors = errors[0:1]

    crops = CROPS_DICT.get(scene_name, [
        Cropbox(top=120, left=120, height=96, width=128, scale=5),
        Cropbox(top=220, left=220, height=96, width=128, scale=5)
    ])

    figure = MyCropComparison(
        reference_image=reference_image,
        method_images=methods_images,
        crops=crops,
        scene_name=ARGS.scene_names_to_show_map[scene_name],
        method_names=method_names_show,
        use_latex=USE_LATEX,
        use_color=True,
        header=header,

        errors_provided=errors,
        error_metrics_provided=metrics,
    )

    return figure


def gen_main_exp(scene_names, only_relMSE):
    ''' Generate the main experiment PDF '''
    if ARGS.one_pdf:
        rows = []
        for scene_name in scene_names:
            figure = generate_figure_one_scene(scene_name, only_relMSE, len(rows) == 0)
            if figure is None:
                print("Skipping scene `{}` due to missing image.".format(scene_name))
                continue
            print("Process Scene `{}` Finished".format(scene_name))
            rows.append(figure.figure_row)
        save_filename = "{}/all-scenes.pdf".format(ARGS.exp_dir)
        figuregen.figure(rows, width_cm=25, filename=save_filename)
        print("All scenes processed. Saved to: {}".format(save_filename))
    else:
        for scene_name in scene_names:
            figure = generate_figure_one_scene(scene_name, only_relMSE)
            save_filename = "{}/{}.pdf".format(ARGS.exp_dir, scene_name)
            figuregen.figure([figure.figure_row], width_cm=25, filename=save_filename)
            print("Process Scene `{}` Finished, save to {}".format(scene_name, save_filename))


##############################################################
##################### [Fig2 Error Curve] #####################
##############################################################

def load_error(scene_name, method_name):
    json_path = "{}/{}_{}.exr.json".format(ARGS.exp_dir, scene_name, method_name)

    y = None
    x = None
    infer_time = None

    with open(json_path, "r") as f:
        data = json5.load(f)
        # RelMSE
        y = [i["RelMSE"] for i in data["data"]]
        # time
        x = data["timepoints"]
        # spp
        # data["timesteps"]
        infer_time = data["infer-time"]

    # get last infer_time seconds
    start_time = x[-1] - infer_time
    skip_seconds = start_time + ARGS.curve_skip_seconds
    x = [i - start_time for i in x if i >= skip_seconds]
    y = y[-len(x):]

    ddx, ddy = [], []
    for i in range(len(y)):
        if i == 0 or (y[i - 1] != y[i] and (not ARGS.curve_clip_error or y[i] < rmin)):
            ddx.append(x[i])
            ddy.append(y[i])
            # ddy.append(y[i] * x[i])
            rmin = y[i]

    return (ddx, ddy)


def generate_curve_image(data, max_x=60):
    assert len(data) <= len(METHOD_COLORS), "Not enough colors defined for the number of methods."

    plot = figuregen.MatplotLinePlot(aspect_ratio=1.0, data=data)
    plot.set_colors(METHOD_COLORS[0:len(data)])

    plot.set_axis_label('x', "Time [s]")
    plot.set_axis_label('y', "")

    min_x = ARGS.curve_skip_seconds
    max_y = max([max(dy) for _, dy in data])
    y_ticks = [0.01, 0.05]
    if (max_y > 0.1):
        y_ticks.append(0.1)

    plot.set_axis_properties('x', ticks=[min_x, 10, 30], range=[min_x, max_x], use_log_scale=False)
    plot.set_axis_properties('y', ticks=y_ticks, range=None, use_log_scale=True)

    return plot


def curve_main_grid(scene_names):
    print("Now Infer Time = 60s")
    infer_time = 60

    images = []
    for scene_name in scene_names:
        print("Process Scene `{}`".format(scene_name))
        data = [load_error(scene_name, method) for method in ARGS.method_names]
        subimage = generate_curve_image(data=data, max_x=infer_time)
        images.append(subimage)

    grid_num = len(scene_names)
    grid_dim_col = 6
    if grid_num % 5 == 0:
        grid_dim_col = 5
    grid_dim_row = (grid_num + grid_dim_col - 1) // grid_dim_col

    num = grid_dim_row * grid_dim_col
    empty = num - grid_num
    if empty > 0:
        # print red error
        print("\033[91mWarning: Not enough grids to display all scenes. Filling with {} empty grids.\033[0m".format(empty))

    # scene_name_prefix = "\\textsc{{{}}}" if USE_LATEX else "{}"
    scene_name_prefix = "{}"
    scene_name_show = [scene_name_prefix.format(ARGS.scene_names_to_show_map[name]) for name in scene_names]
    scene_name_show.extend([""] * empty)  # fill empty grids

    grids = []
    image_idx = 0
    empty_image = figuregen.PNG(np.ones((1, 1, 3), dtype=np.uint8) * 255)  # white placeholder image
    for i in range(grid_dim_row):
        subgrid = figuregen.Grid(1, grid_dim_col)
        for j in range(grid_dim_col):
            if image_idx < len(images):
                subgrid.get_element(0, j).set_image(images[image_idx])
                image_idx += 1
            else:
                # fill empty grid with a placeholder image
                subgrid.get_element(0, j).set_image(empty_image)

        layout = subgrid.layout
        layout.column_space = 1.0
        layout.row_space = 1.0

        # scene names
        plot_titles = scene_name_show[i * grid_dim_col:(i + 1) * grid_dim_col]
        subgrid.set_col_titles(figuregen.BOTTOM, plot_titles)
        layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=3, offset=0.5)

        # metric
        subgrid.set_row_titles(figuregen.LEFT, ["RelMSE"])
        layout.row_titles[figuregen.LEFT] = figuregen.TextFieldLayout(fontsize=8, size=3, offset=0.5, rotation=90)
        layout.set_padding(right=1.0)

        grids.append([subgrid])

    return grids


def curve_caption_grid(scene_num):
    num = len(ARGS.method_names_show)

    empty_image = figuregen.PNG(np.ones((1, 5, 3), dtype=np.uint8) * 255)  # white placeholder image
    pich, picw = empty_image.height_px, empty_image.width_px

    grid = None

    use_label = False

    # [Soluiton 1] put methods name on the right side (use label)
    # not easy to align: this code only works for grid_cols = 7 & figure width_cm = 26.0
    if use_label:
        grid_cols = 7
        grid = figuregen.Grid(1, grid_cols)
        for i in range(grid_cols):
            subgrid = grid.get_element(0, i)
            subgrid.layout.row_space = 1
            subgrid.layout.column_space = 1
            subgrid.set_image(empty_image)

            j = int(math.floor(i - (grid_cols - num) / 2))
            if j < 0 or j >= num:
                continue
            subgrid.draw_lines(start_positions=[[0.5 * pich, 0.1 * picw]],
                               end_positions=[[0.5 * pich, 0.4 * picw]],
                               linewidth_pt=2.0, color=METHOD_COLORS[j])
            name = ARGS.method_names_show[j]
            subgrid.set_label(name, "top_left", offset_mm=[15, 1.5], width_mm=15, fontsize=8)

    # [Solution 2] put methods name on the bottom (use titles)
    else:
        grid_cols = 10 if num % 2 == 0 else 9
        grid_cols += 2

        grid = figuregen.Grid(1, grid_cols)
        for i in range(grid_cols):
            subgrid = grid.get_element(0, i)
            subgrid.layout.row_space = 1
            subgrid.layout.column_space = 1
            subgrid.set_image(empty_image)

        offset = (grid_cols - num) // 2

        # draw lines
        for i in range(num):
            subgrid = grid.get_element(0, i + offset)
            subgrid.draw_lines(start_positions=[[0.2 * empty_image.height_px, 0.3 * empty_image.width_px]],
                               end_positions=[[0.2 * empty_image.height_px, 0.7 * empty_image.width_px]],
                               linewidth_pt=2.0, color=METHOD_COLORS[i])

        # titles
        plot_titles = deepcopy(ARGS.method_names_show)
        plot_titles = ([""] * offset) + plot_titles + ([""] * offset)
        grid.set_col_titles(figuregen.BOTTOM, plot_titles)
        layout = grid.layout
        offset = -1.8 if (scene_num % 5 == 0) else -2.5
        layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=3, offset=offset)
        layout.set_padding(top=1.0)

    return [grid]


def error_curve(scene_names):
    ''' TODO: we don't check file-not-exist errors here,
              because we only generate the result successfully passed the gen_main_exp() function
    '''
    main_grid = curve_main_grid(scene_names)
    caption_grid = curve_caption_grid(len(scene_names))

    main_grid.append(caption_grid)

    save_filename = "{}/all-scenes-curve.pdf".format(ARGS.exp_dir)
    width_cm = 21.0 if len(scene_names) % 5 == 0 else 26.0
    figuregen.figure(main_grid, width_cm=width_cm, filename=save_filename)
    print("All scenes processed. Saved to: {}".format(save_filename))

##############################################################
######################### [Figs End] #########################
##############################################################


def auto_detect_scene_names(directory):
    """
    Automatically detect scene names from the given directory.
    Assumes that the scene names are the prefixes of the image files in the directory.
    """
    scene_names = set()
    for filename in os.listdir(directory):
        if filename.endswith(".exr"):
            ridx = filename.rfind("_")
            scene_name = filename[:ridx] if ridx != -1 else filename
            scene_names.add(scene_name)

    scene_names = list(scene_names)
    # print("Detected scene names: {}".format(scene_names))

    return scene_names


def scene_name_to_show_map():
    scene_names = given_scene_names()

    m = dict()
    for scene_name in scene_names:
        m[scene_name] = scene_name

    m.update({
        "apartment1-v2": "pool",
        "apartment1-v2-view2": "pool-large",
        "m1-f210": "measure-one",
        "noenv_apartment2": "apartment2",
        "sponza-view2": "sponza",
        "sun-temple-v2": "sun-temple-dark",
        "sun-temple-v2-view2": "sun-temple-interior",
        "sun-temple-v2-lighter-view2": "sun-temple-interior",
        "tropical_bedroom_cx": "tropical-bedroom",
        "cornell-box2": "cornell-box",
        "cylinder-2": "cylinder",
        "sponza-caustic-finer3-v2-view2": "sponza-caustic-finer",
        "san-miguel-view2": "san-miguel",
        "sponza-caustic-lowf": "sponza-caustic (smooth)",
    })

    for i in m:
        m[i] = m[i].replace("_", " ").replace("-", " ").title()

    return m


def given_scene_names():
    return [
        # main pics
        "apartment1",
        "apartment3",
        "bistro-interior",
        "noenv_apartment2",
        "sponza-caustic",
        "sun-temple-v2-lighter-view2",

        # sup main pics
        "apartment1-v2-view2",
        "m1-f210",
        "sponza-glossy",
        "sun-temple",
        "tropical_bedroom_cx",
        "san-miguel-view2",
    ]


def scene_names_sup():
    return [
        "apartment1-v2-view2",
        "m1-f210",
        "sponza-glossy",
        "sun-temple",
        "tropical_bedroom_cx",
        "san-miguel-view2",
    ]


def scene_names_main_pics():
    return [
        "apartment1",
        "apartment3",
        "bistro-interior",
        "noenv_apartment2",
        "sponza-caustic",
        "sun-temple-v2-lighter-view2",
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PDF for crop comparisons.")
    parser.add_argument("--one_pdf", action="store_true", help="Generate a single PDF for all scenes.")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum depth for ray tracing.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory for input images.")
    parser.add_argument("--curve_skip_seconds", type=int, default=5, choices=range(0, 11), help="Number of seconds to skip for curve generation.")
    parser.add_argument("--curve_clip_error", action="store_false", help="Clip the errors going up")
    args = parser.parse_args()

    args.exp_dir = "{}/{}".format(CURRENT_DIR, args.image_dir)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # check whether image_dir and max_depth match
    image_dir_base = os.path.basename(args.image_dir)
    print("Image Directory: {}".format(image_dir_base))
    assert args.max_depth == int(image_dir_base.split("-")[0][1:]), \
        "max_depth must match image_dir (e.g., image_dir=d6-xxx, max_depth=6). [{} vs {}]".format(args.max_depth, args.image_dir)

    # reference
    args.reference_dir = os.path.join(PROJECT_DIR, "common", "assets", "scenes-nrrs", "references", "d{}".format(args.max_depth))

    # print all arguments
    print("Arguments:")
    for arg in vars(args):
        print("  {}: {}".format(arg, getattr(args, arg)))
    print("")

    return args


if __name__ == "__main__":
    ARGS = parse_args()

    method_names = ["pt", "adrrs", "ears", "ears+", "adn", "nrrs", "nrrs+", "1#0@nrrs", "1#0@nrrs+"]
    method_names_show = ["PT", "ADRRS", "EARS", "EARS+", "ADRRS (NN)", "NRRS", "NRRS+", "AID-NRRS", "AID-NRRS+"]
    only_relMSE = False

    # ### [1] auto-detect scene names
    # method_names = ["pt", "adrrs", "ears", "adn", "nrrs+", "1#0@nrrs+"]
    # method_names_show = ["PT", "ADRRS", "EARS", "ADRRS (NN)", "NRRS+", "AID-NRRS+"]
    # scene_names = auto_detect_scene_names(ARGS.exp_dir)

    # ### [2] all
    # scene_names = given_scene_names()

    # ### [3] main_pics
    method_names = ["pt", "adrrs", "ears", "adn", "nrrs+", "1#0@nrrs+"]
    method_names_show = ["PT", "ADRRS", "EARS", "ADRRS (NN)", "NRRS+", "AID-NRRS+"]
    # scene_names = scene_names_main_pics()
    scene_names = auto_detect_scene_names(ARGS.exp_dir) # for experiments

    # ### [4] sup main_pics (more)
    # method_names = ["pt", "adrrs", "ears", "adn", "nrrs+", "1#0@nrrs+"]
    # method_names_show = ["PT", "ADRRS", "EARS", "ADRRS (NN)", "NRRS+", "AID-NRRS+"]
    # scene_names = scene_names_sup()

    # ### [5] ablations
    # scene_names = ["bistro-interior", "sun-temple-v2-lighter-view2",]
    # method_names = ['0#0@nrrs', '0#0@nrrs+', '0#1@nrrs', '0#1@nrrs+', '0#2@nrrs', '0#2@nrrs+', '0#3@nrrs', '0#3@nrrs+',
    #                 '1#0@nrrs', '1#0@nrrs+', '1#1@nrrs', '1#1@nrrs+', '1#2@nrrs', '1#2@nrrs+', '1#3@nrrs', '1#3@nrrs+']
    # # METHOD_NAMES filter "nrrs+"
    # method_names = [name for name in method_names if "nrrs+" in name]
    # method_names.insert(0, "pt")
    # method_names_show = method_names.copy()
    # method_names_show = [name.replace("#", "-") for name in method_names_show]

    # ### [6] failure case
    # scene_names = ["cornell-box2"]
    # only_relMSE = True
    # CBOX = True
    # method_names = ["pt", "adrrs", "ears", "adn", "nrrs", "nrrs+", "1#0@nrrs+"]
    # method_names_show = ["PT", "ADRRS", "EARS", "ADRRS (NN)", "NRRS", "NRRS+", "AID-NRRS+"]

    # ### [7] smooth sponza caustic
    # method_names = ["pt", "adrrs", "ears", "adn", "nrrs+", "1#0@nrrs+"]
    # method_names_show = ["PT", "ADRRS", "EARS", "ADRRS (NN)", "NRRS+", "AID-NRRS+"]
    # scene_names = ["sponza-caustic-lowf"]

    # sort by mapped name
    m = scene_name_to_show_map()
    scene_names.sort(key=lambda x: m.get(x, x))

    print("Detected `{}` scene names: {}".format(len(scene_names), scene_names))

    ARGS.scene_names_to_show_map = m
    ARGS.method_names = method_names
    ARGS.method_names_show = method_names_show

    gen_main_exp(scene_names, only_relMSE)
    error_curve(scene_names)
