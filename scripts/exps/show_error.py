# ------------------------------------------
# For development / testing only: add parent directory to python path so we can load the package without installing it
# DO NOT use this if you have installed figuregen via pip
import sys
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(CURRENT_DIR, "figure-gen"))  # root directory
# -------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
from figuregen.util import image
import argparse


def relMSE(img, ref, epsilon=0.01):
    # Compute the relative mean squared error between two images
    ret = ((img - ref) ** 2 / (ref ** 2 + epsilon))
    ret = np.mean(ret, axis=-1)  # Average over the color channels
    return ret


def jet(v, vmax):
    v = max(0, min(v, vmax))
    c = [1.0, 1.0, 1.0]
    if v < (0.25 * vmax):
        c[0] = 0.0
        c[1] = 4 * v / vmax
    elif v < (0.5 * vmax):
        c[0] = 0.0
        c[2] = 1 + 4 * (0.25 * vmax - v) / vmax
    elif v < (0.75 * vmax):
        c[0] = 4 * (v - 0.5 * vmax) / vmax
        c[2] = 0.0
    else:
        c[1] = 1 + 4 * (0.75 * vmax - v) / vmax
        c[2] = 0.0
    return c


def jet_ndarray(arr, jetVMax):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0, jetVMax)
    c = np.ones(arr.shape + (3,), dtype=np.float32)
    mask1 = arr < (0.25 * jetVMax)
    mask2 = (arr >= (0.25 * jetVMax)) & (arr < (0.5 * jetVMax))
    mask3 = (arr >= (0.5 * jetVMax)) & (arr < (0.75 * jetVMax))
    mask4 = arr >= (0.75 * jetVMax)

    c[mask1, 0] = 0.0
    c[mask1, 1] = 4 * arr[mask1] / jetVMax

    c[mask2, 0] = 0.0
    c[mask2, 2] = 1 + 4 * (0.25 * jetVMax - arr[mask2]) / jetVMax

    c[mask3, 0] = 4 * (arr[mask3] - 0.5 * jetVMax) / jetVMax
    c[mask3, 2] = 0.0

    c[mask4, 1] = 1 + 4 * (0.75 * jetVMax - arr[mask4]) / jetVMax
    c[mask4, 2] = 0.0

    return c


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--img", type=str, required=True)
    args.add_argument("--ref", type=str, required=True)
    args = args.parse_args()

    img = cv.imread(args.img, cv.IMREAD_UNCHANGED)[..., 0:3]
    ref = cv.imread(args.ref, cv.IMREAD_UNCHANGED)[..., 0:3]

    jet_max = 0.01

    # cv.imshow("Reference Image", ref)
    error_img = relMSE(img, ref)

    jet_img = jet_ndarray(error_img, jet_max)
    # RGB => BGR
    jet_img = jet_img[..., ::-1]

    print(error_img.flatten().mean())
    cv.imshow("Error Image", jet_img)
    # save error image to exr
    cv.imwrite("error_image.exr", jet_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
