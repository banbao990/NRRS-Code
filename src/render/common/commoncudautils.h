#pragma once
#include "cudahints.h"

// sum: reducton kernel
__global__ void SumTwoPassSimpleKernel(const float *input, float *part_sum, const uint32_t n);
__global__ void SumTwoPassSimpleKernelOp1(const float *input, float *part_sum, const uint32_t n);
__global__ void SumTwoPassInterleavedKernel(const float *input, float *part_sum, uint32_t n);

template <bool rcpSum>
__global__ void SumTwoPassInterleavedKernelOp1(const float *input, float *part_sum, size_t n);
__global__ void SumSimple(const float *input, float *sum, size_t n);

// utils
__device__ inline float RRSWeightWindow(const float splittingFactor,
										const float weightWindowSize = 5) {
	const float dminus = 2 / (1 + weightWindowSize);
	const float dplus  = dminus * weightWindowSize;

	if (splittingFactor < dminus) {
		/// russian roulette
		return splittingFactor / dminus;
	} else if (splittingFactor > dplus) {
		/// splitting
		return splittingFactor / dplus;
	} else {
		/// within weight window
		return 1.0f;
	}
}