import math
import os
import subprocess
import json5
import time
import json
from copy import deepcopy
import argparse

from configs import ALL_SCENES
import random


parser = argparse.ArgumentParser(description="Render Reference Experiments")
parser.add_argument("--max_depth", type=int, required=True, help="Max depth")
parser.add_argument("--spp", type=int, default=200000, help="Samples per pixel (spp) for rendering")
args = parser.parse_args()


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_PATH, "..")

SCENE_PATH_PREFIX = "common/configs/nrrs/scenes/"

scene_lists = deepcopy(ALL_SCENES)

# reverse
# scene_lists = list(reversed(scene_lists))


METHOD_CONFIG_PATH = os.path.join(BASE_PATH, "common/configs/nrrs/render/render-ref.json")
SCENE_CONFIG_PATH = os.path.join(BASE_PATH, "common/configs/nrrs/scenes/empty.json")

CMD = "{} -method {} -scene {}".format(
    os.path.join(BASE_PATH, "out/build/x64-Release/src/testbed.exe"),
    METHOD_CONFIG_PATH, SCENE_CONFIG_PATH
)


def check_global_name():
    # check [1] scene_lists have "global" keyword
    #       [2] global have "name" keyword
    for scene in scene_lists:
        scene_config = os.path.join(BASE_PATH, SCENE_PATH_PREFIX, scene)
        with open(scene_config, "r") as f:
            data = json5.load(f)
            # f: scene_name.json
            scene_name = scene.split(".")[0]
            ok = False
            if "global" in data:
                d = data["global"]
                if "name" in d and d["name"] == scene_name:
                    ok = True
            if (not ok):
                print(f"scene {scene} not found global name")
                return False
    return True


def run_command(command):
    print("Running command: {}".format(command))
    process = subprocess.Popen(command, shell=False)
    while process.poll() is None:
        time.sleep(1)

    return process.returncode


def update_scene_config(scene_path):
    with open(SCENE_CONFIG_PATH, "r") as f:
        data = json5.load(f)

    data["scene_path"] = scene_path

    # write back
    with open(SCENE_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)


def d6_minus_d10():
    ''' return scene list those have d6 refs but not d10 refs'''
    ref_dirs = os.path.join(BASE_PATH, "common/assets/scenes-nrrs/references/")

    d6_refs = os.listdir(os.path.join(ref_dirs, "d6"))
    d6_refs = [f[0:f.rfind("_")] for f in d6_refs]
    d6_refs = set(d6_refs)

    d10_refs = os.listdir(os.path.join(ref_dirs, "d10"))
    d10_refs = [f[0:f.rfind("_")] for f in d10_refs]
    d10_refs = set(d10_refs)

    return ["{}.json".format(i) for i in list(d6_refs - d10_refs)]


if __name__ == "__main__":
    if args.max_depth == 10:
        input("Are you sure detect the loss refs automatically?")
        # scene_lists = d6_minus_d10()
        print(f"found {len(scene_lists)} scenes have d6 refs but not d10 refs")
    print(scene_lists)

    try:
        input("you should render low spp to ensure scene config is ok!")

        if (not check_global_name()):
            # red color error
            print("\033[31m[ERROR] global name not found\033[0m")
            exit(1)

        # update max_depth
        with open(METHOD_CONFIG_PATH, "r") as f:
            data = json5.load(f)

        random_offset = random.randint(0, 1000000)  # can add multi-image
        data["passes"][0]["params"]["max_depth"] = args.max_depth
        data["passes"][0]["params"]["random_offset"] = random_offset
        data["passes"][1]["params"]["task"]["value"] = args.spp
        data["passes"][1]["params"]["save_every"] = min(max((args.spp // 10), 10), 40000)

        with open(METHOD_CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=4)
        for scene in scene_lists:
            update_scene_config(os.path.join(SCENE_PATH_PREFIX, scene))
            while (run_command(CMD) != 0):
                print("Re-running due to error...")
    except KeyboardInterrupt:
        pass
    finally:
        # reset scene config
        update_scene_config("")
