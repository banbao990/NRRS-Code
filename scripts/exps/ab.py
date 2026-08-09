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
from copy import deepcopy

from main_exps import scene_name_to_show_map, scene_names_main_pics
from utils import ORDER_COLORS_LATEX


def load_error(dir: str, scene_name: str, method_name: str, silent=False):
    error_file = os.path.join(dir, "{}_{}.exr.json".format(scene_name, method_name))
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            data = json5.load(f)
            return data["data"][-1]["RelMSE"]
    else:
        # red error string and exit()
        if not silent:
            print("\033[91m[Error]: file `{}` not found.\033[0m".format(error_file))
        return math.nan


def rank_asc(arr):
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])
    ranks = [0] * len(arr)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank
    return ranks


def mix_depth(max_depth):
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))

    error_dir = os.path.join(CURRENT_DIR, "images", "exps", "d{}-final".format(max_depth))
    method_names = ['pt', 'nrrs', 'nrrs+', '1#0@nrrs', '1#0@nrrs+', 'ears', 'ears+']
    colors = deepcopy(ORDER_COLORS_LATEX)
    # use last color for padding
    colors = colors + [colors[-1]] * (len(method_names) - len(colors))
    for scene_name in scene_names:
        error_scene = []
        for method_name in method_names:
            error_data = load_error(error_dir, scene_name, method_name)
            error_scene.append(error_data)

        error_strs = []
        for error in error_scene:
            error_strs.append("{:.2f}~({:.2f}$\\times$)".format(1e3 * error, error / error_scene[0]))

        idxs = rank_asc(error_scene)

        # color
        error_strs = ["\\fboxsep1pt\\colorbox{}{{{}}}".format(colors[idx], error_str) for idx, error_str in zip(idxs, error_strs)]

        print(scene2show[scene_name], "&", "&".join(error_strs), "\\\\")
    print()


def statnet():
    scene2show = scene_name_to_show_map()
    scene_names = [
        "bistro-interior",
        "sun-temple-v2-lighter-view2",  # only 1 with above
    ]

    headers = ["NRRS", "NRRS+", "AID-NRRS", "AID-NRRS+"]
    cols = ["All", "PartNet", "Encoding", "None"]
    cols_type = [0, 1, 2, 3]

    error_dir = os.path.join(CURRENT_DIR, "images", "exps", "d6-ab-final")

    best_color = ORDER_COLORS_LATEX[0]
    for scene_idx, scene_name in enumerate(scene_names):
        # calculate errors
        errors_scene = []
        for col_idx, col in enumerate(cols):
            errors_dir_col = []
            for method_type in headers:
                method_name = "{}#{}@nrrs{}".format(
                    "0" if method_type.find("AID") == -1 else "1",
                    cols_type[col_idx],
                    "+" if method_type.find("+") != -1 else ""
                )
                error = load_error(error_dir, scene_name, method_name)
                errors_dir_col.append(error)
            errors_scene.append(errors_dir_col)

        # argmin
        errors_scene = np.array(errors_scene)
        min_indices = np.argmin(errors_scene, axis=0)

        # output
        if scene_idx == 0:
            print("\\toprule")
            print("&", "&".join(headers), "\\\\")
        print("\\midrule")
        print("\\multicolumn{{{}}}{{l}}{{\\textbf{{{}}}}} \\\\".format(len(cols) + 1, scene2show[scene_name]))
        for col_idx, col in enumerate(cols):
            print(col, "&", end="")
            errors_str = []
            for method_type_idx, _ in enumerate(headers):
                v = 1e3 * errors_scene[col_idx][method_type_idx]
                # min RelMSE
                if col_idx == min_indices[method_type_idx]:
                    error_str_prefix = "\\fboxsep1pt\\colorbox{}{{{:.2f}}}".format(best_color, v)
                else:
                    error_str_prefix = "{:.2f}".format(v)
                errors_str.append(error_str_prefix)
            print("&".join(errors_str), "\\\\")
        if scene_idx == len(scene_names) - 1:
            print("\\bottomrule")


def ablation_1():
    scene2show = scene_name_to_show_map()
    scene_names = scene_names_main_pics()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    dirs = ["d6-final", "d6-ab-final/d6-noavg", "d6-ab-final/d6-denoise-always", "d6-ab-final/d6-denoise-none"]
    error_dirs = [os.path.join(CURRENT_DIR, "images", "exps", d) for d in dirs]
    method_names = ['nrrs', 'nrrs', 'nrrs_denoise_always', "nrrs"]
    method_names_show = ['NRRS', 'no $\mathcal{L}_{\\text{avg}}$', 'Denoise Always', 'No Denoise']

    print("\\begin{{tabular}}{{{}}}".format(("c" * (len(method_names_show) + 1))))
    print("\\toprule")
    print("&", "&".join(method_names_show), "\\\\")
    print("\\midrule")
    for scene_name in scene_names:
        error_scene = []
        for method_idx, method_name in enumerate(method_names):
            error_data = load_error(error_dirs[method_idx], scene_name, method_name, True)
            error_scene.append(error_data)

        error_strs = []
        for error in error_scene:
            error_strs.append("{:.3f}".format(1e3 * error))

        print(scene2show[scene_name], "&", "&".join(error_strs), "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def earsnn():
    scene2show = scene_name_to_show_map()
    scene_names = scene_names_main_pics()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    error_dir = os.path.join(CURRENT_DIR, "images", "exps", "d6-final")
    method_names = ['ears', 'earsn', 'earst']
    method_names_show = ['EARS', 'EARS (NN)', 'EARS (NT)']

    print("\\begin{{tabular}}{{{}}}".format(("c" * (len(method_names_show) + 1))))
    print("\\toprule")
    print("&", "&".join(method_names_show), "\\\\")
    print("\\midrule")
    for scene_name in scene_names:
        error_scene = []
        for method_idx, method_name in enumerate(method_names):
            error_data = load_error(error_dir, scene_name, method_name, True)
            error_scene.append(error_data)

        error_strs = []
        for error in error_scene:
            error_strs.append("{:.3f}".format(1e3 * error))

        print(scene2show[scene_name], "&", "&".join(error_strs), "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def print_main_table(max_depth):
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))

    error_dir = os.path.join(CURRENT_DIR, "images", "exps", "d{}-final".format(max_depth))
    method_names = ["pt", "adrrs", "ears", "adn", "nrrs+", "1#0@nrrs+"]
    method_names_show = ["PT", "ADRRS", "EARS", "ADRRS (NN)", "NRRS+", "AID-NRRS+"]
    colors = deepcopy(ORDER_COLORS_LATEX)

    print("\\begin{{tabular}}{{{}}}".format(("c" * (len(method_names_show) + 1))))
    print("\\toprule")
    print("Scene&", "&".join(method_names_show), "\\\\")
    print("\\midrule")
    # use last color for padding
    colors = colors + [colors[-1]] * (len(method_names) - len(colors))
    for scene_name in scene_names:
        error_scene = []
        for method_name in method_names:
            error_data = load_error(error_dir, scene_name, method_name)
            error_scene.append(error_data)

        error_strs = []
        for error in error_scene:
            error_strs.append("{:.2f}~({:.2f}$\\times$)".format(1e3 * error, error / error_scene[0]))

        idxs = rank_asc(error_scene)

        # color
        error_strs = ["\\fboxsep1pt\\colorbox{}{{{}}}".format(colors[idx], error_str) for idx, error_str in zip(idxs, error_strs)]

        print(scene2show[scene_name], "&", "&".join(error_strs), "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def show_denoise(simple):
    from error_pics import jet_ndarray

    dir = os.path.join(CURRENT_DIR, "images", "exps", "denoise")
    files = ["error_origin.exr", "error_denoise.exr"]
    names = ["Before OptiX Denoiser", "After OptiX Denoiser"]
    jet_max = 5
    num = len(files)
    imgs = [cv.imread(os.path.join(dir, f), cv.IMREAD_UNCHANGED)[:, :, 0] for f in files]
    names = ["{} (mean={:.2f})".format(name, np.mean(img)) for name, img in zip(names, imgs)]
    jet_imgs = [jet_ndarray(img, jet_max)for img in imgs]
    del imgs

    split_grid = [figuregen.Grid(num_rows=1, num_cols=1) for _ in range(num)]
    for i in range(num):
        split_grid[i][0, 0].set_image(figuregen.PNG(jet_imgs[i]))
    for grid_idx, grid in enumerate(split_grid):
        if (grid_idx != num - 1):
            # last grid do not need padding
            grid.layout.padding[figuregen.RIGHT] = 1
        if simple:
            grid.set_col_titles(figuregen.BOTTOM, names[grid_idx:grid_idx + 1])
            grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=3, offset=0.5)

    # tint
    from split import load_image
    scale = 1.4
    tint_ori = load_image("images/exps/tint/tint-vertical.png")
    tint = np.ones((tint_ori.shape[0], int(tint_ori.shape[1] * scale), tint_ori.shape[2]), dtype=np.float32)
    offset = tint_ori.shape[1] // (6 if simple else 10)
    w = tint.shape[1] // 2
    tint[:, w - offset: w + offset, :] = tint_ori[:, 0:1, :]

    tint_grid = figuregen.Grid(num_rows=1, num_cols=1)
    tint_grid[0, 0].set_image(figuregen.PNG(tint))
    tint_grid.set_col_titles(figuregen.BOTTOM, [str(0)])
    tint_grid.set_col_titles(figuregen.TOP, [str(jet_max)])
    size = 4 if simple else 10
    tint_grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=size, offset=0)
    tint_grid.layout.column_titles[figuregen.TOP] = figuregen.TextFieldLayout(fontsize=8, size=size, offset=0)

    if simple:
        filename = os.path.join(dir, "denoise-ba-simple.pdf")
        figuregen.figure([split_grid + [tint_grid]], width_cm=10, filename=filename)
        print("Save to {}".format(filename))
        return

    # crops in main images
    crop_colors_deep = [
        [255, 255, 255]
    ]
    from main_exps import CROPS_DICT
    crops = CROPS_DICT["bistro-interior"]
    crops_num = len(crops)

    for i in range(num):
        for j in range(crops_num):
            crop = crops[j]
            split_grid[i][0, 0].set_marker(crop.marker_pos, crop.marker_size, color=crop_colors_deep[j % len(crop_colors_deep)])
    # crop grid
    crop_grid = [
        figuregen.Grid(num_cols=crops_num, num_rows=1)
        for _ in range(num)
    ]
    for method_idx in range(num):
        crop = crop_grid[method_idx]
        crop.layout.padding[figuregen.TOP] = 1
        if (method_idx != num - 1):
            # last grid do not need padding
            crop.layout.padding[figuregen.RIGHT] = 1
        for crop_idx in range(crops_num):
            crop[0, crop_idx].image = figuregen.PNG(crops[crop_idx].crop(jet_imgs[method_idx]))
            crop[0, crop_idx].set_frame(linewidth=1, color=crop_colors_deep[crop_idx % len(crop_colors_deep)])

        crop.set_title(figuregen.BOTTOM, names[method_idx])
        crop.layout.titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=3, offset=0.5)

    filename = os.path.join(dir, "denoise-ba.pdf")
    figuregen.figure([split_grid, crop_grid], width_cm=10, filename=filename)

    # load pdf, add tint
    img_without_tint = figuregen.PDF(filename)
    grid = figuregen.Grid(1, 1)
    grid.get_element(0, 0).set_image(img_without_tint)
    figuregen.figure([[grid, tint_grid]], width_cm=10, filename=filename)

    print("Save to {}".format(filename))


def ablations_34():

    scene2show = scene_name_to_show_map()
    scene_names = scene_names_main_pics()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))

    # NRRS
    dirs = ["d6-final"]
    method_names = ['nrrs']
    method_names_show = ['NRRS']

    # denoise design
    dirs += ["d6-ab-final/d6-denoise-always", "d6-ab-final/d6-denoise-none"]
    method_names += ['nrrs_denoise_always', 'nrrs']
    method_names_show += ['Denoise Always', 'No Denoise']

    # EARS with denoised pixel estimate
    dirs += ["d6-ab-final/d6-denoisedI"]
    method_names += ['ears_denoisedI']
    method_names_show += ['\\BBAdd{EARS (D)}']

    # EARS variants
    dirs += ["d6-final", "d6-final", "d6-final"]
    method_names += ['ears', 'earsn', 'earst']
    method_names_show += ['EARS', 'EARS (NN)', "EARS (NT)"]

    # loss design
    dirs += ["d6-ab-final/d6-noavg", "d6-ab-final/d6-bound=1"]
    method_names += ['nrrs', 'nrrs']
    method_names_show += ['$-\mathcal{L}_{\\text{avg}}$', '\\BBAdd{$+\mathcal{L}_{\\text{defensive}}$}']

    error_dirs = [os.path.join(CURRENT_DIR, "images", "exps", d) for d in dirs]

    segs = [0, 1, 3, 4, 7]
    alignment = ("c" * (len(method_names_show) + 1))
    # add '|' after each seg
    for seg_idx in range(len(segs)):
        seg_v = segs[seg_idx] + 1  # +1 for before '|'
        alignment = alignment[:seg_v + seg_idx] + "|" + alignment[seg_v + seg_idx:]

    print("\\begin{{tabular}}{{{}}}".format(alignment))
    print("\\toprule")
    print("&", "&".join(method_names_show), "\\\\")
    print("\\midrule")
    for scene_name in scene_names:
        error_scene = []
        for method_idx, method_name in enumerate(method_names):
            error_data = load_error(error_dirs[method_idx], scene_name, method_name, True)
            error_scene.append(error_data)

        error_strs = []
        errors = []
        for error in error_scene:
            error_strs.append("{:.2f}".format(1e3 * error))
            errors.append(error if not math.isnan(error) else float('inf'))
        # color the lowest one
        min_idx = np.argmin(errors)
        error_strs[min_idx] = "\\fboxsep1pt\\colorbox{}{{{}}}".format(ORDER_COLORS_LATEX[0], error_strs[min_idx])

        # BBAdd [for revision]
        error_strs[3] = "\\BBAdd{{{}}}".format(error_strs[3])
        error_strs[-1] = "\\BBAdd{{{}}}".format(error_strs[-1])

        print(scene2show[scene_name], "&", "&".join(error_strs), "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def sup_denoisedI():

    scene2show = scene_name_to_show_map()
    scene_names = scene_names_main_pics()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))

    # PT should be the first one for rate calculation
    # PT, NRRS
    dirs = ["d6-final"] * 2
    method_names = ['pt', 'nrrs']
    method_names_show = ['PT', 'NRRS']

    # ADRRS
    dirs += ["d6-final", "d6-ab-final/d6-denoisedI"]
    method_names += ['adrrs', 'adrrs_denoisedI']
    method_names_show += ['ADRRS', 'ADRRS (D)']

    # EARS
    dirs += ["d6-final", "d6-ab-final/d6-denoisedI"]
    method_names += ['ears', 'ears_denoisedI']
    method_names_show += ['EARS', 'EARS (D)']

    error_dirs = [os.path.join(CURRENT_DIR, "images", "exps", d) for d in dirs]

    segs = [0, 1, 2, 4]
    alignment = ("c" * (len(method_names_show) + 1))
    # add '|' after each seg
    for seg_idx in range(len(segs)):
        seg_v = segs[seg_idx] + 1  # +1 for before '|'
        alignment = alignment[:seg_v + seg_idx] + "|" + alignment[seg_v + seg_idx:]

    print("\\begin{{tabular}}{{{}}}".format(alignment))
    print("\\toprule")
    print("&", "&".join(method_names_show), "\\\\")
    print("\\midrule")
    for scene_name in scene_names:
        error_scene = []
        for method_idx, method_name in enumerate(method_names):
            error_data = load_error(error_dirs[method_idx], scene_name, method_name, True)
            error_scene.append(error_data)

        error_strs = []
        errors = []
        for error in error_scene:
            error_strs.append("{:.2f}".format(1e3 * error))
            errors.append(error if not math.isnan(error) else float('inf'))
        # color the lowest one
        min_idx = np.argmin(errors)

        # add rates
        error_strs = [e + "~({:.2f}$\\times$)".format(error / error_scene[0]) for e, error in zip(error_strs, error_scene)]

        error_strs[min_idx] = "\\fboxsep1pt\\colorbox{}{{{}}}".format(ORDER_COLORS_LATEX[0], error_strs[min_idx])

        print(scene2show[scene_name], "&", "&".join(error_strs), "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def load_spp(dir: str, scene_name: str, method_name: str):
    error_file = os.path.join(dir, "{}_{}.exr.json".format(scene_name, method_name))
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            data = json5.load(f)
            timepoints = data["timepoints"]
            timesteps = data["timesteps"]
            total_spp = timesteps[-1]
            # find idx where timepoint ~ 60s
            timepoints = np.array(timepoints)
            idx_after = np.searchsorted(timepoints, 60.0)
            idx_before = idx_after - 1
            # linear interpolation to find spp at 60s
            t_before = timepoints[idx_before]
            t_after = timepoints[idx_after]
            spp_before = timesteps[idx_before]
            spp_after = timesteps[idx_after]
            spp_train = spp_before + (spp_after - spp_before) * (60.0 - t_before) / (t_after - t_before)

            spp_train = int(spp_train)
            spp_infer = int(total_spp - spp_train)

            # print("Scene `{}` Method `{}`: total spp = {}, spp train = {}, spp infer = {}".format(
            #     scene_name, method_name, total_spp, spp_train, spp_infer))
            # print(timepoints[idx_before], timepoints[idx_after], timesteps[idx_before], timesteps[idx_after])
            return spp_train, spp_infer

    else:
        print("\033[91m[Error]: file `{}` not found.\033[0m".format(error_file))
        return 0, 0


def spp_train_infer():

    scene2show = scene_name_to_show_map()
    scene_names = scene_names_main_pics()

    scene_names.sort(key=lambda x: scene2show.get(x, x))
    scene_names = scene_names_main_pics()
    scene2show = scene_name_to_show_map()

    scene_names.sort(key=lambda x: scene2show.get(x, x))

    # ### config start

    # NRRS
    data_dir = os.path.join(CURRENT_DIR, "images", "exps", "d6-final")
    method_names = ['nrrs', 'ears']
    method_names_show = ['NRRS', 'EARS']

    # ### config end

    methods_count = len(method_names)
    alignment = "c" + ("|cc" * (len(method_names_show)))

    print("\\begin{{tabular}}{{{}}}".format(alignment))
    print("\\toprule")

    header_parts = ["\\multirow{2}{*}{Scene}"]
    for i, method_name in enumerate(method_names_show):
        seg_after = "" if i == methods_count - 1 else "|"
        header_parts.append("& \\multicolumn{{2}}{{c{}}}{{{}}} ".format(seg_after, method_name))  # format need {{, }}
    print(" ".join(header_parts), "\\\\")
    header_parts = []
    for i in range(methods_count):
        seg_after = "" if i == methods_count - 1 else "|"
        header_parts.append("& \\multicolumn{1}{c}{training} & \\multicolumn{1}{c" + seg_after + "}{inference} ")
    print(" ".join(header_parts), "\\\\")

    print("\\midrule")
    for scene_name in scene_names:
        print(scene2show[scene_name], end=" ")
        for method_name in method_names:
            spp_train, spp_infer = load_spp(
                data_dir,
                scene_name,
                method_name
            )
            print("&{}".format(spp_train), end=" ")
            print("&{}".format(spp_infer), end=" ")
        print("\\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    # ### [1] Mix-Depth
    # mix_depth(6)
    # mix_depth(10)

    # ### [2] StatNet Type
    # statnet()

    # ### [3] no L_avg & DenoiseAlways & DenoiseNone
    # ablation_1()

    # ### [4] ears-nn
    # earsnn()

    # ### [3&4]
    # ablations_34()

    # ### [5] print d10 main [in fact not ablation]
    # print_main_table(10)

    # ### [6] show denoise
    # show_denoise(True)
    # show_denoise(False)

    # ### [7] spp train and infer
    # spp_train_infer()

    # ### [8] denoisedI [sup]
    # sup_denoisedI()

    pass
