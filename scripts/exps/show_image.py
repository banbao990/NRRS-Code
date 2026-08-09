# ------------------------------------------
# For development / testing only: add parent directory to python path so we can load the package without installing it
# DO NOT use this if you have installed figuregen via pip
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(CURRENT_DIR, "figure-gen"))  # root directory
# -------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
from figuregen.util import image
import argparse

args = argparse.ArgumentParser()
args.add_argument("--path", type=str, required=True)
args.add_argument("--gamma", action="store_true")
args = args.parse_args()


img = cv.imread(args.path, cv.IMREAD_UNCHANGED)[..., 0:3]
# img = img[:, :, ::-1]

# tonemapping
img = image.lin_to_srgb(img)
if args.gamma:
    img = img**0.45454545

cv.imshow("Tonemapped Image", img)
cv.waitKey(0)
cv.destroyAllWindows()

# cv.imwrite(args.path + ".out.png", img * 255)
