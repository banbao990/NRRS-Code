#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "device/cuda.h"

NAMESPACE_BEGIN(krr)

// [ATTENTION] partSum should have enough size
template <bool tRcpSum>
void calcSum2PassAsync(const float *input, float *sum, float *partSum, size_t size,
					   CUstream stream);

struct StatMinMaxAvgGPU {
public:
	KRR_DEVICE void init() {
		mMin   = 1e10f;
		mMax   = -1e10f;
		mAvg   = 0.0f;
		mCount = 0;
		initLock();
	}

	KRR_DEVICE void initLock() { mLock = 0u; }
	KRR_DEVICE void acquireLock() {
		while (!atomicCAS(&mLock, 0u, 1u)) {
			continue;
		}
	}
	KRR_DEVICE void releaseLock() { atomicCAS(&mLock, 1u, 0u); }

	KRR_DEVICE void record(float v) {
		acquireLock();
		mMin = min(mMin, v);
		mMax = max(mMax, v);
		mAvg += v;
		mCount++;
		releaseLock();
	}

	KRR_DEVICE void print() { printf("min: %f, max: %f, avg: %f\n", mMin, mMax, mAvg / mCount); }

private:
	uint mLock;
	float mMin;
	float mMax;
	float mAvg;
	uint mCount;
};

union Float2Int {
	float f;
	int i;
};

NAMESPACE_END(krr)