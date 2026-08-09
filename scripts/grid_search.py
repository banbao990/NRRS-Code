import os
import subprocess
import time
import json5
import json
import random
import numpy as np
from itertools import product

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_PATH, "..")

GRID_SEARCH_DIR = os.path.join(BASE_PATH, "common/outputs/test/test/grid_search")

if (os.path.exists(GRID_SEARCH_DIR) == False):
    os.makedirs(GRID_SEARCH_DIR)

TIME_TO_KILL = 60


def run_command(command):
    start = time.time()
    process = subprocess.Popen(command, shell=False)
    while process.poll() is None:
        time.sleep(1)
        if time.time() - start > TIME_TO_KILL:
            process.kill()
            return None
    return process.returncode


def update_json(gamma1, gamma2, gamma3, gamma4=None):
    SRC_JSON = os.path.join(BASE_PATH, "common/configs/nrrs/nrrs/2net-simple.json")
    DST_JSON = os.path.join(BASE_PATH, "common/configs/nrrs/nrrs/2net-simple-test.json")

    # read json
    data = None
    with open(SRC_JSON, "r") as f:
        data = json5.load(f)

    # update json
    if (gamma1 is not None):
        data["nn"]["level1"]["loss"]["gamma1"] = gamma1
    if (gamma2 is not None):                
        data["nn"]["level1"]["loss"]["gamma2"] = gamma2
    if (gamma3 is not None):                
        data["nn"]["level1"]["loss"]["gamma3"] = gamma3
    if (gamma4 is not None):               
        data["nn"]["level1"]["loss"]["gamma4"] = gamma4

    # save json
    with open(DST_JSON, "w") as f:
        json.dump(data, f, indent=4)


def get_time_stamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def check_file_exists(file_name) -> bool:
    files = os.listdir(GRID_SEARCH_DIR)
    for file in files:
        if file_name in file:
            return True
    return False


def grid_search124():
    input("RUNNING grid_search124()")

    # grid search
    bounds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    steps = len(bounds)

    # total steps
    print("Total steps: {}".format(steps ** 3))
    print("Total time: {} h".format(steps**3 * TIME_TO_KILL / 60 / 60))

    t1_str = get_time_stamp()
    t1 = time.time()
    cnt = 0

    params = list(product(bounds, repeat=3))

    # random_shuffle
    random.shuffle(params)

    # get gamma4
    SRC_JSON = os.path.join(BASE_PATH, "common/configs/nrrs/nrrs/2net-simple.json")
    # read json
    data = None
    with open(SRC_JSON, "r") as f:
        data = json5.load(f)

    # gamma3 = data["nn"]["level1"]["loss"]["gamma3"]
    # now gamma3 is fixed to 0.0
    gamma3 = 0.0

    ignore_exist = []
    for ggg in params:
        gamma1, gamma2, gamma4 = ggg
        file_name = "{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(gamma1, gamma2, gamma3, gamma4)
        existed = check_file_exists(file_name)
        if existed:
            ignore_exist.append([gamma1, gamma2, gamma4])

    print("configs: {}".format(len(params)))

    ignore_exist = list(set(map(tuple, ignore_exist)))
    params = list(set(map(tuple, params)) - set(map(tuple, ignore_exist)))

    print("configs (after remove exists): {}".format(len(params)))

    for ggg in params:
        gamma1, gamma2, gamma4 = ggg

        print("gamma1: {}, gamma2: {}, gamma3: {}, gamma4: {}".format(gamma1, gamma2, gamma3, gamma4))

        file_name = "{}_{}_{}_{}".format(gamma1, gamma2, gamma3, gamma4)
        update_json(gamma1, gamma2, gamma3, gamma4)

        # run command
        run_command(CMD)

        cnt += 1

    # time.sleep(5)
    t2_str = get_time_stamp()
    t2 = time.time()
    print("{} -> {}".format(t1_str, t2_str))
    print("total time: {:.4f} s = {:.4f} h".format(t2 - t1, (t2 - t1) / 60 / 60))

    # 2024-12-25-23-38-14 -> 2024-12-26-09-28-44
    # total time: 35429.3175 s = 9.8415 h


def grid_search4():
    input("RUNNING grid_search4()")

    TIME_TO_KILL = 80
    bounds = np.linspace(0.1, 5, 50)
    bounds = np.append(bounds, 0)
    steps = len(bounds)

    # total steps
    print("Total steps: {}".format(steps))
    print("Total time: {} h".format(steps * TIME_TO_KILL / 60 / 60))

    t1_str = get_time_stamp()
    t1 = time.time()
    cnt = 0

    skip_params = []

    random.shuffle(bounds)

    # get gamma1,2,3
    SRC_JSON = os.path.join(BASE_PATH, "common/configs/nrrs/nrrs/2net-simple.json")
    # read json
    data = None
    with open(SRC_JSON, "r") as f:
        data = json5.load(f)
    g1 = data["nn"]["level1"]["loss"]["gamma1"]
    g2 = data["nn"]["level1"]["loss"]["gamma2"]
    g3 = data["nn"]["level1"]["loss"]["gamma3"]

    for g4 in bounds:
        print("gamma4: {}".format(g4))

        file_name = "{}_{}_{}_{}".format(g1, g2, g3, g4)
        existed = check_file_exists(file_name)
        if existed:
            print("skip: {}".format(file_name))
            skip_params.append(file_name)

        else:
            update_json(None, None, None, g4)

            # run command
            run_command(CMD)

        cnt += 1

    # time.sleep(5)
    t2_str = get_time_stamp()
    t2 = time.time()
    print("{} -> {}".format(t1_str, t2_str))
    print("total time: {:.4f} s = {:.4f} h".format(t2 - t1, (t2 - t1) / 60 / 60))
    print("skip params: {}".format(skip_params))


def show_topk(k, dir=GRID_SEARCH_DIR):
    files = os.listdir(dir)
    files = map(lambda x: x.split("_")[0:-1], files)
    files = map(lambda x: list(map(float, x)), files)
    files = sorted(files, key=lambda x: [x[0], x[1]])
    for i, file in enumerate(files):
        if k > 0 and i >= k:
            break
        print(file)


def check_define(file_name):
    ok = True
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "#define BB_TCNN_DEBUG_MODE" in line:
                if "//" in line:
                    ok = True
                    print("debug mode is disabled")
                else:
                    ok = False
                    print("debug mode is enabled")
                break

    if (ok):
        # printf green info
        print("\033[32m check {} passed [-Wformat]\033[0m".format(file_name))
        return
    # printf red warning
    print("\033[31m check {} failed [-Wformat]\033[0m".format(file_name))
    exit(-1)


def check_config():
    TEST_JSON = os.path.join(BASE_PATH, "common/configs/nrrs/render/test.json")
    with open(TEST_JSON, "r") as f:
        data = json5.load(f)

    nrrs_config = data["passes"][0]["params"]
    config1 = nrrs_config["nrrs_config"]
    if (config1.endswith("test.json")):
        pass
    else:
        print("config is '{}'".format(config1))
        exit(-1)

    aid_mode = nrrs_config["net1_aid_net2"]
    input("AID mode is \033[32m{}\033[0m. Are you sure? (y/n)".format(aid_mode))

    time_controls = [
        nrrs_config["nrrs_auto_train_state1_time"],
        nrrs_config["nrrs_auto_train_state2_time"],
        nrrs_config["nrrs_auto_train_state3_time"]
    ]

    print()
    input("Are you sure with these time config?\ntime controls: {}\n".format(time_controls))


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(BASE_PATH, "common/outputs/test/test")

    if (os.path.exists(OUTPUT_DIR) == False):
        os.makedirs(OUTPUT_DIR)

    CMD = "{} {}".format(
        os.path.join(BASE_PATH, "out/build/x64-Release/src/testbed.exe"),
        os.path.join(BASE_PATH, "common/configs/render/test.json")
    )

    NRRS_H = os.path.join(BASE_PATH, "src/ext/tcnn/include/tiny-cuda-nn/losses/nrrs.h")
    check_define(NRRS_H)
    check_config()

    grid_search124()

    # grid_search4()

    # show_topk(10, GRID_SEARCH_DIR + "_0104")
