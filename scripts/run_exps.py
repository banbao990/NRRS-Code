import os
import subprocess
import json5
import time
import json
import argparse
import random


from configs import ALL_SCENES

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_PATH, "..")

parser = argparse.ArgumentParser(description="Run Experiments")
# ray counter(bool)
parser.add_argument("--rc", action="store_true", help="Use ray counter")
parser.add_argument("--max_depth", type=int, required=True, help="Max depth")
args = parser.parse_args()


# METHOD_CONFIG_PATH = os.path.join(BASE_PATH, "common/configs/nrrs/render/render-ref.json")
SCENE_CONFIG_PATH = os.path.join(BASE_PATH, "common/configs/nrrs/scenes/empty.json")

TRAIN_TIME = 60
INFERENCE_TIME = 60

KILL_TIME = TRAIN_TIME + INFERENCE_TIME + 120

EXPS_OUTPUT_PATH = os.path.join(BASE_PATH, "common/exps")

if not os.path.exists(EXPS_OUTPUT_PATH):
    os.makedirs(EXPS_OUTPUT_PATH)


def run_command(command):
    start = time.time()
    process = subprocess.Popen(command, shell=False)
    while process.poll() is None:
        time.sleep(1)
        if time.time() - start > KILL_TIME:
            process.kill()
            return None
    return process.returncode


def update_scene_config(scene_path):
    with open(SCENE_CONFIG_PATH, "r") as f:
        data = json5.load(f)

    data["scene_path"] = scene_path
    data["global"] = {"reference_dir": "common/assets/scenes-nrrs/references/"}

    # write back
    with open(SCENE_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)


def get_command(method):
    cmd = "{} -method {} -scene {}".format(
        os.path.join(BASE_PATH, "out/build/x64-Release/src/testbed.exe"),
        method, SCENE_CONFIG_PATH
    )
    return cmd


def test_scenes(scene_config_list):
    global INFERENCE_TIME
    inference_time = INFERENCE_TIME

    INFERENCE_TIME = 5

    method_name_list = ["pt"]
    run_all_methods(method_name_list, scene_config_list)

    INFERENCE_TIME = inference_time


def run_all_methods(method_name_list, scene_config_list, indices=None, method_first=False):
    scene_name_lists = gen_scene_name_from_config(scene_config_list)

    scene_num = len(scene_name_lists)
    method_num = len(method_name_list)

    print("{} Scenes: {}".format(scene_num, scene_name_lists))
    print("{} Methods: {}".format(method_num, method_name_list))

    method_list = gen_method_config_from_name(method_name_list)

    if indices is None:
        if method_first:
            indices = [(m, s) for m in range(method_num) for s in range(scene_num)]
        else:
            indices = [(m, s) for s in range(scene_num) for m in range(method_num)]

    print("{} Indices".format(len(indices)))

    # METHOD_LISTS x SCENE_LISTS
    for method_idx, scene_idx in indices:
        method = os.path.join(BASE_PATH, method_list[method_idx])
        scene = scene_config_list[scene_idx]
        update_scene_config(scene)

        # update method config
        with open(method, "r") as f:
            data = json5.load(f)

        method_name = method_name_list[method_idx]

        # global config
        exp_dir = os.path.join(EXPS_OUTPUT_PATH, str(args.max_depth))
        if (not os.path.exists(exp_dir)):
            os.makedirs(exp_dir)
        exp_output_file = os.path.join(exp_dir, "{}_{}.exr".format(scene_name_lists[scene_idx], method_name))
        # if (os.path.exists(exp_output_file)):
        #     print("Experiment output file already exists: {}".format(exp_output_file))
        #     continue

        denoisedI = method_name.find("denoisedI") != -1

        denoise_always = False
        denoise_always_idx = method_name.find("_denoise_always")
        if denoise_always_idx != -1:
            denoise_always = True
            method_name = method_name[:denoise_always_idx]

        data["global"]["exp_output_file"] = exp_output_file
        data["global"]["exp_train_time"] = TRAIN_TIME
        data["global"]["exp_inference_time"] = INFERENCE_TIME
        data["global"]["exp_on"] = True
        data["global"]["exp_denoise_always"] = denoise_always

        # pass config
        data["passes"][0]["params"]["rrs_method"] = method_name.split("@")[-1]
        data["passes"][0]["params"]["max_depth"] = args.max_depth

        if (method_name.find("nrrs") != -1 and method_name.find("@") != -1):
            # ablation
            abalation = method_name.split("@")[0].split("#")
            net1_aid_net2 = True if abalation[0] == "1" else False
            two_head_type = int(abalation[1])
            data["passes"][0]["params"]["net1_aid_net2"] = net1_aid_net2
            data["passes"][0]["params"]["ll2_use_two_head"] = two_head_type != 0
            data["passes"][0]["params"]["ll2_two_head_type"] = two_head_type

        if denoisedI:
            data["passes"][0]["params"]["use_acc_buffer_not_denoised_nrrs"] = False  # for nrrs
            data["passes"][0]["params"]["use_acc_buffer_not_denoised_ears"] = False  # for ears/adrrs

        while (not os.path.exists(exp_output_file)):
            # for stability
            random_offset = random.randint(0, 1000000)
            data["passes"][0]["params"]["random_offset"] = random_offset
            data["passes"][1]["params"]["random_offset"] = random_offset  # errormeasure pass
            with open(method, "w") as f:
                json.dump(data, f, indent=4)

            run_command(get_command(method))
            # exit()


def gen_scene_name_from_config(scene_list):
    scene_name_list = []
    for scene in scene_list:
        scene_name = os.path.basename(scene).split(".")[0]
        scene_name_list.append(scene_name)
    return scene_name_list


def gen_scene_configs():
    scene_list = set(ALL_SCENES)

    # main scenes
    scene_list = [
        "apartment1",
        "apartment3",
        "bistro-interior",
        "noenv_apartment2",
        "sponza-caustic",
        "sun-temple-v2-lighter-view2",
    ]

    # supplementary scenes
    scene_list = [
        "apartment1-v2-view2",
        "m1-f210",
        "sponza-glossy",
        "sun-temple",
        "tropical_bedroom_cx",
        "san-miguel-view2",
    ]

    # test scenes
    scene_list = [
        "noenv_apartment2",
    ]

    scene_list.sort()

    scene_list = ["common/configs/nrrs/scenes/{}.json".format(scene) for scene in scene_list]

    return scene_list


def gen_method_config_from_name(method_name_list):
    name2method_map = {
        "pt": "wpt",
        "adrrs": "adrrs_ears",
        "ears": "adrrs_ears",
        "ears+": "adrrs_ears",
        "earsn": "adrrs_ears",
        "ears_denoisedI": "adrrs_ears",
        "adrrs_denoisedI": "adrrs_ears",
        "earst": "adrrs_ears",  # nn + tree
        "adn": "adn_nrrs",
        "nrrs": "adn_nrrs",
    }

    method_list = [(i if i.find("nrrs") == -1 else "nrrs") for i in method_name_list]
    method_list = ["common/configs/nrrs/render/exps/{}.json".format(name2method_map[m]) for m in method_list]
    return method_list


def main_methods():
    method_name_list = [
        "pt",
        "adrrs",
        "ears",
        # "ears_denoisedI",
        # "adrrs_denoisedI",
        # "ears+",
        # "earsn",
        "adn",
        # "nrrs",
        "nrrs+",
        # "1#0@nrrs",
        "1#0@nrrs+",

        # "nrrs_denoise_always",
    ]

    # method_name_list = [
    # "nrrs_denoise_always",
    # "nrrs",
    # ]
    return method_name_list


def ablation_aid_and_statnet():

    method_name_list = [
        # aid
        # "1#0@nrrs", "1#0@nrrs+",
    ]

    # ablations for aid x StatNet
    for aid in range(2):
        for two_head_type in range(4):
            method_name_list.append("{}#{}@nrrs".format(aid, two_head_type))
            method_name_list.append("{}#{}@nrrs+".format(aid, two_head_type))

    return method_name_list


def given_pairs():
    ''' for stability '''

    todo = [

    ]

    scenes = []
    methods = []
    for item in todo:
        scene, method = item.split(",")
        scenes.append("common/configs/nrrs/scenes/{}.json".format(scene))
        methods.append(method)
    indices = [(i, i) for i in range(len(scenes))]
    return methods, scenes, indices


if __name__ == "__main__":
    try:

        method_name_list = main_methods()
        # method_name_list = ablation_aid_and_statnet()

        # method_name_list.extend(ablation_aid_and_statnet())

        scene_config_list = gen_scene_configs()

        # should update "exp_output_file", "rrs_method"

        # first run all simple pt to make sure the scene is correct
        # test_scenes(scene_config_list)

        # run not ok methods
        # pairs = given_pairs()
        # run_all_methods(*pairs, False)

        # run all methods
        run_all_methods(method_name_list, scene_config_list, None, False)

    except KeyboardInterrupt as e:
        pass
    finally:

        # reset scene config
        update_scene_config("")
