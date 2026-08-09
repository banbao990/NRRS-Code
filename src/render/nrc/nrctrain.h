/*This file should only be included in CUDA cpp files*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "device/atomic.h"
#include "device/cuda.h"
#include "workitem.h"
#include "workqueue.h"

#include <tiny-cuda-nn/common.h>

NAMESPACE_BEGIN(krr)

template <typename T> class DeviceBuffer {
public:
	DeviceBuffer() = default;

	KRR_HOST DeviceBuffer(int n) : mMaxSize(n) { cudaMalloc(&mData, n * sizeof(T)); }

	KRR_CALLABLE int push(const T &w) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		(*this)[index % mMaxSize] = w;
		return index;
	}

	KRR_CALLABLE void clear() { mSize.store(0); }

	KRR_CALLABLE int size() const { return mSize.load(); }

	KRR_CALLABLE T *data() { return mData; }

	KRR_CALLABLE T &operator[](int index) {
		DCHECK_LT(index, mMaxSize);
		return mData[index];
	}

	KRR_CALLABLE DeviceBuffer &operator=(const DeviceBuffer &w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

protected:
	KRR_CALLABLE int allocateEntry() { return mSize.fetch_add(1); }

private:
	atomic<int> mSize;
	T *mData;
	int mMaxSize{0};
};

template <typename NetworkInput, typename NetworkOutput> class NetworkTrainBuffer {
public:
	NetworkTrainBuffer() = default;

	KRR_HOST NetworkTrainBuffer(int n) : mMaxSize(n) {
		cudaMalloc(&mInputs, n * sizeof(NetworkInput));
		cudaMalloc(&mOutputs, n * sizeof(NetworkOutput));
	}

	KRR_CALLABLE int push(const NetworkInput &input, const NetworkOutput &output) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		mInputs[index]	= input;
		mOutputs[index] = output;
		return index;
	}

	KRR_CALLABLE void clear() { mSize.store(0); }

	KRR_CALLABLE int size() const {
#ifndef KRR_DEVICE_CODE
		CUDA_SYNC_CHECK();
		cudaDeviceSynchronize();
#endif
		return mSize.load();
	}

	KRR_CALLABLE void resize(int n) {
		if (mMaxSize) {
			cudaFree(mInputs);
			cudaFree(mOutputs);
		}
		cudaMalloc(&mInputs, n * sizeof(NetworkInput));
		cudaMalloc(&mOutputs, n * sizeof(NetworkOutput));
	}

	KRR_CALLABLE NetworkInput *inputs() const { return mInputs; }

	KRR_CALLABLE NetworkOutput *outputs() const { return mOutputs; }

	KRR_CALLABLE NetworkTrainBuffer &operator=(const NetworkTrainBuffer &w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

private:
	KRR_CALLABLE int allocateEntry() { return mSize.fetch_add(1); }

	atomic<int> mSize;
	NetworkInput *mInputs;
	NetworkOutput *mOutputs;
	int mMaxSize{0};
};

KRR_CALLABLE Vector3f normalizeSpatialCoord(const Vector3f &coord, AABB aabb) {
	constexpr float inflation = 0.005f;
	aabb.inflate(aabb.diagonal().norm() * inflation);
	return Vector3f{0.5} + (coord - aabb.center()) / aabb.diagonal();
}

KRR_CALLABLE float nrc_warp_roughness_for_ob(const float roughness) { return 1 - expf(-roughness); }

__global__ void
nrc_generate_training_data(const size_t nElements, uint trainPixelOffset, uint trainPixelStride,
						   NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput> *trainBuffer,
						   NRCPathPixelStateBuffer *guidedState, const AABB sceneAABB);

__global__ void nrc_generate_inference_data(const size_t nElements,
											NRCInferenceQueue *mScatterRayQueue, float *data,
											const AABB sceneAABB);

NAMESPACE_END(krr)