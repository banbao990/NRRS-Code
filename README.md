# README

> [中文版](README.CN.md)

+ Official implementation of the TVCG 2026 paper [NRRS: Neural Russian Roulette and Splitting](https://ieeexplore.ieee.org/document/11512998)
+ Project website: [NRRS](https://banbao990.github.io/publications/NRRS/)



## Download

```bash
git clone https://github.com/banbao990/NRRS-Code.git
```

```bash
git submodule update --init --recursive --force
```



## Tested Environment

+ Windows 10
+ MSVC 2022, v17.14.29
+ CUDA 12.5.40
+ Vulkan 1.3.238
+ RTX 3080
+ OptiX 8.0.0



## Build and Run

+ Environment variables
    + Set `OptiX_INSTALL_DIR` to the OptiX installation directory.
    + Make sure the CUDA directory is configured correctly when installing CUDA.
+ Open the entire folder as a CMake project in MSVC and select `x64-Release: RelWithDebInfo`.
    + Parallel build: Project -> CMake Settings -> set the build option to `-j 4`.
+ Run `testbed.exe`.
    + Because the test scene is large, it has been split into smaller files. Reassemble it first by running `scripts\restore_scene.bat`.

+ Training starts automatically after launching:
    + With the automatic training configuration, StatNet is trained for 10 seconds, followed by RRSNet training.
    + Click `Stop Training` to stop training manually.
    + Toggle `Enable RRS` to enable or disable RRS and observe the results of the method presented in the paper.
+ Default configuration:
    + Render graph: [test.json](common/configs/nrrs/render/test.json)
    + Scene: [test.json](common/configs/nrrs/scenes/test.json)



## Experiment Scripts

+ After the build is complete, run the following command from the repository root to reproduce the main experiments:
    + [Main experiment script](scripts/run_exps.py). **If an error curve contains an outlier—for example, a single frame causes the error to spike abnormally—rerun the experiment.**
    + Python 3.10

+ Use this [script](scripts/render_ref.py) to render reference images.
+ The [exps](scripts/exps) directory contains the plotting scripts used for the paper.



### Example

+ Run the experiment script on the test scene:

```bash
python scripts/run_exps.py --max_depth=6
```

+ Output directory: `common/exps/6`
+ Copy the output files to `scripts/exps/images/d6-test`.
    + The directory name must begin with `d6` or `d10` to indicate the maximum path depth.
    + this example uses `d6-test`.
+ Run the plotting script:

```bash
python scripts/exps/main_exps.py --one_pdf --max_depth 6 --curve_skip_seconds 5 --image_dir images/d6-test
```

+ The plotting script produces:
    + A grid comparing the different experimental methods.
    + Error curves for the different methods, which can be inspected for anomalies.



## Core Code

+ The complete integrator is implemented in the [nrrs](src/render/nrrs) directory.
    + This includes the entire wavefront framework, the RRS normalization module, and the network training and inference logic.
+ The network loss consists of two parts:
    + [StatNet Loss](src/ext/tcnn/include/tiny-cuda-nn/losses/nrrs_ll2.h)
    + [RRSNet Loss](src/ext/tcnn/include/tiny-cuda-nn/losses/nrrs_rrs.h)



## TCNN Issue

+ The following issue was observed on August 8, 2026:
    + If you do not encounter this issue, you can revert the change described below. The current version uses `/4`.

+ On an RTX 3080, startup may be very slow, with the application appearing to hang on the UI for approximately 120 seconds.
+ This appears to be an issue with tiny-cuda-nn.
+ In `include/tiny-CUDA-nn/gpu_memory.h`, the third call to `cuMemAddressReserve` at line 442 hangs.
+ The current workaround is to reduce the amount of GPU memory requested at line 433:

```diff
--- m_max_size = previous_multiple(cuda_memory_info().total, cuda_memory_granularity());
+++ m_max_size = previous_multiple(cuda_memory_info().total/4, cuda_memory_granularity());
```



## Acknowledgments

+ This project is built on [KiRaRay](https://github.com/cuteday/KiRaRay). We thank [cuteday](https://github.com/cuteday) for open-sourcing this excellent rendering framework.
+ The neural network components are implemented using [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
+ The plotting scripts are implemented using [figure-gen](https://github.com/Mira-13/figure-gen).
