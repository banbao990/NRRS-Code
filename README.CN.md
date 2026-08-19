# README

>  [English](README.md)

+ TVCG 2026 论文 [NRRS: Neural Russian Roulette and Splitting](https://ieeexplore.ieee.org/document/11512998) 官方实现
+ 项目网站：[NRRS](https://banbao990.github.io/publications/NRRS/)



## 下载

```bash
git clone https://github.com/banbao990/NRRS-Code.git
```

```bash
git submodule update --init --recursive --force
```



## 测试系统

+ Windows 10
+ MSVC 2022, v17.14.29
+ CUDA 12.5.40
+ Vulkan 1.3.238
+ RTX 3080
+ OptiX 8.0.0



## 编译运行

+ 环境变量
    + 需要把 OptiX 的安装路径添加到 `OptiX_INSTALL_DIR` 中
    + 安装 CUDA 时正确设置 CUDA 目录
+ 使用 MSVC 将整个文件夹作为 CMake 工程打开，选择 `x64-Release: RelWithDebInfo` 即可
    + 并行编译：项目 -> CMake 设置：build 选项 `-j 4`
+ 运行 `testbed.exe`
    + 测试场景因为有点大，因此需要拆分成小文件了，需要先合并一下；调用 `scripts\restore_scene.bat`

+ 点击运行之后会自动启动
    + 进入自动训练的配置：训练 StatNet 10s，然后进入到 RRSNet 的训练
    + 手动点击 `Stop Training` 结束训练
    + `Enable RRS` 开启关闭 RRS，能看到论文方法的实验效果
+ 默认配置
    + render graph 文件：[test.json](common/configs/nrrs/render/test.json)
    + 场景文件：[test.json](common/configs/nrrs/scenes/test.json)



## 测试脚本

+ 编译完成之后，在根目录运行如下命令，就可以进行主实验的测试
    + [主实验测试脚本](scripts/run_exps.py)【注意如果某条误差曲线出现异常，例如某一帧导致误差异常升高，需要重新跑】
    + python 3.10

+ 渲染 reference [脚本](scripts/render_ref.py)
+ [exps](scripts/exps) 文件夹下为论文绘图脚本



### 例子

+ 调用实验脚本跑测试场景

```bash
python scripts/run_exps.py --max_depth=6
```

+ 文件输出目录：`common/exps/6`
+ 将文件拷贝到文件夹 `scripts/exps/images/d6-test` 下
    + 注意文件夹名称必须以 `d6/d10` 开头【表示最大深度】，这里用的是 `d6-test`
+ 运行绘图脚本

```bash
python scripts/exps/main_exps.py  --one_pdf --max_depth 6 --curve_skip_seconds 5 --image_dir images/d6-test
```

+ 绘图脚本输出
    + 一个对比不同实验方法的框图
    + 一个不同方法的误差曲线【可以看是否存在异常】



## 核心代码

+ 整体 integrator 实现在文件夹 [nrrs](src/render/nrrs) 中
    + 包括整个 Wavefront 框架、RRS 归一化模块、网络的训练、推理逻辑
+ 网络 loss 包括两个部分
    + [StatNet Loss](src/ext/tcnn/include/tiny-cuda-nn/losses/nrrs_ll2.h)
    + [RRSNet Loss](src/ext/tcnn/include/tiny-cuda-nn/losses/nrrs_rrs.h)



## TCNN 问题

+ 2026.08.08 测试发现如下问题
    + 如果测试没有问题的话，可以修改回去【目前我用了 `/4`】

+ 在 RTX 3080 测试发现，启动的时候可能会非常慢，表现为卡在 UI 界面约 120s
+ tiny-CUDA-nn 的问题
+ `include/tiny-CUDA-nn/gpu_memory.h` 文件行 442 在第三次调用 `cuMemAddressReserve` 发生卡顿
+ 目前简单修复策略，减小申请的显存大小，行 433

```diff
--- m_max_size = previous_multiple(cuda_memory_info().total, cuda_memory_granularity());
+++ m_max_size = previous_multiple(cuda_memory_info().total/4, cuda_memory_granularity());
```



## 致谢

+ 本项目基于 [KiRaRay](https://github.com/cuteday/KiRaRay) 开发，感谢 [cuteday](https://github.com/cuteday) 开源这一优秀的渲染框架。
+ 神经网络部分基于 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 实现。
+ 绘图脚本基于 [figure-gen](https://github.com/Mira-13/figure-gen) 实现。
