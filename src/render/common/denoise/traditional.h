#pragma once
#include "../cudahints.h"

__global__ void GaussianFilter(const int nElements, const int size, const int width,
							   const int height, const float *color, float *output);

__global__ void BilateralFilter(const int nElements, const int size, const int width,
								const int height, const float *color, const float *normal,
								float *output);