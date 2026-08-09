#include "traditional.h"
#include "common.h"
#include <render/color.h>

__device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator+=(float3 &a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
__device__ float3 operator/=(float3 &a, float b) {
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}

__global__ void GaussianFilter(const int nElements, const int size, const int width,
							   const int height, const float *color, float *output) {

	// color and output are RGB channels
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) return;

	int x = tid % width;
	int y = tid / width;

	int idx			= y * width + x;
	float3 sum		= make_float3(0.0f, 0.0f, 0.0f);
	float weightSum = 0.0f;
	for (int i = -size; i <= size; ++i) {
		for (int j = -size; j <= size; ++j) {
			int x_offset = x + i;
			int y_offset = y + j;
			if (x_offset < 0 || x_offset >= width || y_offset < 0 || y_offset >= height) continue;
			int idx_offset = y_offset * width + x_offset;
			float weight   = expf(-(i * i + j * j) / (2.0f * size * size));

			sum += make_float3(color[idx_offset * 3], color[idx_offset * 3 + 1],
							   color[idx_offset * 3 + 2]) *
				   weight;
			weightSum += weight;
		}
	}

	sum.x /= weightSum;

	output[idx * 3]		= sum.x;
	output[idx * 3 + 1] = sum.y;
	output[idx * 3 + 2] = sum.z;
}

__global__ void BilateralFilter(const int nElements, const int size, const int width,
								const int height, const float *color, const float *normal,
								float *output) {
	// color, normal, output are RGB channels
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) return;
	int x				= tid % width;
	int y				= tid / width;
	int idx				= y * width + x;
	float3 sum			= make_float3(0.0f, 0.0f, 0.0f);
	float weightSum		= 0.0f;
	float3 normalCenter = make_float3(normal[idx * 3], normal[idx * 3 + 1], normal[idx * 3 + 2]);

	for (int i = -size; i <= size; ++i) {
		for (int j = -size; j <= size; ++j) {
			int x_offset = x + i;
			int y_offset = y + j;
			if (x_offset < 0 || x_offset >= width || y_offset < 0 || y_offset >= height) continue;
			int idx_offset		= y_offset * width + x_offset;
			float3 normalOffset = make_float3(normal[idx_offset * 3], normal[idx_offset * 3 + 1],
											  normal[idx_offset * 3 + 2]);
			float weight		= expf(-(i * i + j * j) / (2.0f * size * size)) *
						   expf(-(dot(normalCenter, normalOffset)) / (2.0f * size * size));
			sum += make_float3(color[idx_offset * 3], color[idx_offset * 3 + 1],
							   color[idx_offset * 3 + 2]) *
				   weight;
			weightSum += weight;
		}
	}

	sum.x /= weightSum;
	sum.y /= weightSum;
	sum.z /= weightSum;
	output[idx * 3]		= sum.x;
	output[idx * 3 + 1] = sum.y;
	output[idx * 3 + 2] = sum.z;
}