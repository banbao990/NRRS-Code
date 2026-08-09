#include "commoncudautils.h"

#include <stdio.h>

__global__ void SumTwoPassSimpleKernel(const float *input, float *part_sum, const uint32_t n) {
	// n is divided to gridDim.x part
	const uint32_t num_per_block = ((n + (gridDim.x - 1)) / gridDim.x);
	uint32_t blk_begin			 = num_per_block * blockIdx.x;
	uint32_t blk_end			 = num_per_block * (blockIdx.x + 1);
	blk_end						 = min(blk_end, n);

	// this block process input[blk_begin: blk_end), store result to part_sum
	const uint32_t blk_n   = blk_end - blk_begin;
	const float *input_lbk = input + (uint64_t) blk_begin;
	part_sum += blockIdx.x;

	// n is divided to blockDim.x part
	const uint32_t num_per_thread = ((blk_n + blockDim.x - 1) / blockDim.x);
	uint32_t thr_begin			  = num_per_thread * threadIdx.x;
	uint32_t thr_end			  = num_per_thread * (threadIdx.x + 1);
	thr_end						  = min(thr_end, blk_n);

	// this thread process input_lbk[thr_begin: thr_end)
	float thr_sum = 0.0f;
	for (uint32_t i = thr_begin; i < thr_end; ++i) {
		thr_sum += input_lbk[i];
	}

	// store thr_sum to shared memory
	extern __shared__ float shm[];
	shm[threadIdx.x] = thr_sum;

	// sync threads in this block
	__syncthreads();

	// reduce shm to part_sum
	if (threadIdx.x == 0) {
		float sum = 0.0f;
		for (uint32_t i = 0; i < blockDim.x; ++i) {
			sum += shm[i];
		}
		*part_sum = sum;
	}
}

__global__ void SumTwoPassSimpleKernelOp1(const float *input, float *part_sum, const uint32_t n) {
	// n is divided to gridDim.x part
	const uint32_t num_per_block = ((n + (gridDim.x - 1)) / gridDim.x);
	uint32_t blk_begin			 = num_per_block * blockIdx.x;
	uint32_t blk_end			 = num_per_block * (blockIdx.x + 1);
	blk_end						 = min(blk_end, n);

	// this block process input[blk_begin: blk_end), store result to part_sum
	const uint32_t blk_n   = blk_end - blk_begin;
	const float *input_lbk = input + (uint64_t) blk_begin;
	part_sum += blockIdx.x;

	// n is divided to blockDim.x part
	const uint32_t num_per_thread = ((blk_n + blockDim.x - 1) / blockDim.x);
	uint32_t thr_begin			  = num_per_thread * threadIdx.x;
	uint32_t thr_end			  = num_per_thread * (threadIdx.x + 1);
	thr_end						  = min(thr_end, blk_n);

	// this thread process input_lbk[thr_begin: thr_end)
	float thr_sum = 0.0f;
	for (uint32_t i = thr_begin; i < thr_end; ++i) {
		thr_sum += input_lbk[i];
	}

	// store thr_sum to shared memory
	extern __shared__ float shm[];
	shm[threadIdx.x] = thr_sum;
	// sync threads in this block
	__syncthreads();

	// reduce shm to part_sum
	// we use cudaOccupancyMaxPotentialBlockSize() to get the GridDim, so it must be the 32x
	// __syncthreads() is not necessary here, because they are in the same warp(32)
	if (threadIdx.x < 32) {
		const int turns = blockDim.x / 32 - 1;
		for (int i = 0; i < turns; ++i) {
			shm[threadIdx.x] += shm[threadIdx.x + 32 * (i + 1)];
		}
		for (int32_t active_thread_num = 16; active_thread_num >= 1; active_thread_num /= 2) {
			if (threadIdx.x < active_thread_num) {
				shm[threadIdx.x] += shm[threadIdx.x + active_thread_num];
			}
			__syncwarp();
		}
		if (threadIdx.x == 0) {
			part_sum[blockIdx.x] = shm[0];
		}
	}
}

__global__ void SumTwoPassInterleavedKernel(const float *input, float *part_sum, uint32_t n) {
	// global thread index
	int32_t tid				 = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t total_thread_num = gridDim.x * blockDim.x;

	// reduce
	//   input[tid + total_thread_num * 0]
	//   input[tid + total_thread_num * 1]
	//   input[tid + total_thread_num * 2]
	//   input[tid + total_thread_num * ...]

	float sum = 0.0f;
	for (int32_t i = tid; i < n; i += total_thread_num) {
		sum += input[i];
	}

	// store sum to shared memory
	extern __shared__ float shm[];
	shm[threadIdx.x] = sum;
	__syncthreads();
	// reduce shm to part_sum
	if (threadIdx.x == 0) {
		float sum = 0.0f;
		for (size_t i = 0; i < blockDim.x; ++i) {
			sum += shm[i];
		}
		part_sum[blockIdx.x] = sum;
	}
}

__global__ void SumSimple(const float *input, float *sum, size_t n) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*sum = 0.0f;
	}

	const int index	 = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	float lsum		 = 0.0f;
	for (int i = index; i < n; i += stride) {
		lsum += input[i];
	}
	atomicAdd(sum, lsum);
}

// make sure the template function to be instantiated
template __global__ void SumTwoPassInterleavedKernelOp1<true>(const float *, float *, size_t);
template __global__ void SumTwoPassInterleavedKernelOp1<false>(const float *, float *, size_t);

template <bool rcpSum>
__global__ void SumTwoPassInterleavedKernelOp1(const float *input, float *part_sum, size_t n) {
	// global thread index
	int32_t tid				 = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t total_thread_num = gridDim.x * blockDim.x;

	float sum = 0.0f;
	for (int32_t i = tid; i < n; i += total_thread_num) {
		sum += input[i];
	}

	// store sum to shared memory
	extern __shared__ float shm[];
	shm[threadIdx.x] = sum;
	__syncthreads();

	// reduce shm to part_sum
	// we use cudaOccupancyMaxPotentialBlockSize() to get the GridDim, so it must be the 32x
	// __syncthreads() is not necessary here, because they are in the same warp(32)
	if (threadIdx.x < 32) {
		const int turns = blockDim.x / 32 - 1;
		for (int i = 0; i < turns; ++i) {
			shm[threadIdx.x] += shm[threadIdx.x + 32 * (i + 1)];
		}
		for (int32_t active_thread_num = 16; active_thread_num >= 1; active_thread_num /= 2) {
			if (threadIdx.x < active_thread_num) {
				shm[threadIdx.x] += shm[threadIdx.x + active_thread_num];
			}
			__syncwarp();
		}
		if (threadIdx.x == 0) {
			if (rcpSum) {
				part_sum[blockIdx.x] = 1.0f / shm[0];
			} else {
				part_sum[blockIdx.x] = shm[0];
			}
		}
	}
}