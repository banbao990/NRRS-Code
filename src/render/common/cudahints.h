#pragma once
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#ifndef __CUDACC__
#define __CUDACC__
#endif
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#include <stdint.h>
#include <cuda_runtime.h>