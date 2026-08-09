#pragma once

// data type
#define DATATYPE float
// min number
#define NEG_INF -999999999
#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define GPU_BLOCKS_THRESHOLD 2048
#define GPU_THREADS_THRESHOLD 1024
#define GPU_SHARED_MEM_THRESHOLD 48 * 1024
#define GPU_THREADS 128

#include <cuda_runtime.h>
#include <cuda.h>

void handleCudaError(cudaError_t err, const char *file, int line);
// gpu top k
void top_k_gpu(DATATYPE *input, int length, int k, DATATYPE *output, CUstream stream);
