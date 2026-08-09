# ------------------------------------------
# For development / testing only: add parent directory to python path so we can load the package without installing it
# DO NOT use this if you have installed figuregen via pip
import sys
import os
import json5
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(CURRENT_DIR, "figure-gen"))  # root directory
# -------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
from figuregen import figuregen

from figuregen.util.image import lin_to_srgb
import argparse

from show_error import relMSE, jet_ndarray
from main_exps import image_read, scene_name_to_show_map
from split import load_image

PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))


def gen_row_main(imgs, scene_name, methods_show, tint_img, tint_range):
    img_num = len(imgs)
    pic_grid = figuregen.Grid(num_rows=1, num_cols=img_num)
    for i in range(img_num):
        pic_grid[0, i].set_image(figuregen.PNG(imgs[i]))
    pic_grid.set_col_titles(figuregen.BOTTOM, methods_show)
    # pic_grid.set_col_titles(figuregen.TOP, [""] * img_num)
    pic_grid.set_row_titles(figuregen.LEFT, [scene_name])
    pic_grid.layout.padding[figuregen.RIGHT] = 1
    pic_grid.layout.row_titles[figuregen.LEFT].offset = 0.5
    pic_grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=2.5, offset=0.5)

    tint_grid = figuregen.Grid(num_rows=1, num_cols=1)
    tint_grid[0, 0].set_image(figuregen.PNG(tint_img))

    t = [i * 1e3 for i in tint_range]
    prefix = "{:.1f}" if t[1] < 1e2 else "{:.0f}"
    tint_grid.set_col_titles(figuregen.BOTTOM, [prefix.format(t[0]).rstrip('0').rstrip('.')])
    tint_grid.set_col_titles(figuregen.TOP, [prefix.format(t[1]).rstrip('0').rstrip('.')])

    tint_grid.layout.column_titles[figuregen.TOP] = figuregen.TextFieldLayout(fontsize=8, size=3.5, offset=0)
    tint_grid.layout.column_titles[figuregen.BOTTOM] = figuregen.TextFieldLayout(fontsize=8, size=5.5, offset=0)

    return [pic_grid, tint_grid]


def gen_pics_main(scene_names, image_dir, tint_img, ref_dir, output_name, is_supplementary=False):
    methods = ["ears", "nrrs+"]  # Do Not Change This
    methods_show = ["EARS", "NRRS+"]
    figures = []

    scene_show_name_map = scene_name_to_show_map()
    scene_names.sort(key=lambda x: scene_show_name_map.get(x, x))

    if is_supplementary:
        methods_show.insert(0, "Reference")

    for scene_name in scene_names:
        # check refs
        ref_path = "{}/{}_200000.exr".format(ref_dir, scene_name)
        ref = image_read(ref_path, True)
        jet_imgs = []

        jet_max = 1e-4

        for method in methods:
            img_path = "{}/{}_{}.exr.json".format(image_dir, scene_name, method)
            with open(img_path, "r") as f:
                data = json5.load(f)
                error = data["data"][-1]["RelMSE"]
                jet_max = max(jet_max, error * 1.5)

        for method in methods:
            img_path = "{}/{}_{}.exr".format(image_dir, scene_name, method)
            img = image_read(img_path)

            if img is None:
                # red error
                print("\033[31mImage not found: {}\033[0m".format(img_path))
                exit(-1)

            # Compute error images
            error_img = relMSE(img, ref)
            # print(error_img.flatten().mean())
            jet_img = jet_ndarray(error_img, jet_max)

            # RGB => BGR (opencv is BGR by default)
            # cv.imshow(methods_show[method_idx], jet_img[..., ::-1])
            jet_imgs.append(jet_img)

        scene_name_to_show = scene_show_name_map.get(scene_name, scene_name)
        if is_supplementary:
            ref = lin_to_srgb(ref)  # tone mapping after error calculation
            jet_imgs.insert(0, ref)
        figure = gen_row_main(jet_imgs, scene_name_to_show, methods_show, tint_img, (0, jet_max))
        figures.append(figure)

    width_cm = 20 if is_supplementary else 10
    figuregen.figure(figures, width_cm=width_cm, filename=output_name)
    print("Saved to: {}\n".format(filename))

    # cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--img_dir", type=str, required=True)
    args.add_argument("--max_depth", type=int, required=True)
    args = args.parse_args()

    # check depth match
    image_dir_base = os.path.basename(args.img_dir)
    assert args.max_depth == int(image_dir_base.split("-")[0][1:]), \
        "max_depth must match image_dir (e.g., image_dir=d6-xxx, max_depth=6). [{} vs {}]".format(args.max_depth, args.image_dir)

    scene_names = [
        "bistro-interior",
        "sun-temple-v2-lighter-view2",
        "apartment1",
        "apartment3",
        "noenv_apartment2",
        "sponza-caustic",
    ]

    ref_dir = os.path.join(PROJECT_DIR, "common", "assets", "scenes-nrrs", "references", "d{}".format(args.max_depth))

    # tint
    scale = 1.4
    tint_ori = load_image("images/exps/tint/tint-vertical.png")
    tint = np.ones((tint_ori.shape[0], int(tint_ori.shape[1] * scale), tint_ori.shape[2]), dtype=np.float32)
    offset = tint_ori.shape[1] // 6
    w = tint.shape[1] // 2
    tint[:, w - offset: w + offset, :] = tint_ori[:, 0:1, :]

    # main pics
    # if args.max_depth == 6:
    #     filename = os.path.join(CURRENT_DIR, "images", "exps", "d{}-main-errors.pdf".format(args.max_depth))
    #     gen_pics_main(scene_names[0:2], args.img_dir, tint, ref_dir, filename)
    #     pass

    start_idx = 0 if args.max_depth != 6 else 2
    # supplementary pics
    filename = os.path.join(CURRENT_DIR, "images", "exps", "d{}-supp-errors.pdf".format(args.max_depth))
    gen_pics_main(scene_names[start_idx:], args.img_dir, tint, ref_dir, filename, True)
