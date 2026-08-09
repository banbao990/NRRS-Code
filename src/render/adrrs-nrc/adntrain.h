/*This file should only be included in CUDA cpp files*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "device/atomic.h"
#include "device/cuda.h"
#include "device/gpustd.h"
#include "workitem.h"
#include "workqueue.h"
#include "render/nrc/nrctrain.h"

#include <tiny-cuda-nn/common.h>

NAMESPACE_BEGIN(krr)

template class NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput>;

__global__ void adn_generate_inference_data(const size_t nElements, ADNInferenceQueue *inferQueue,
											float *data, const AABB sceneAABB);
__global__ void
adn_generate_training_data(const size_t nElements,
						   NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput> *trainBuffer,
						   NRCPathPixelStateBuffer *guidedState, const AABB sceneAABB);
NAMESPACE_END(krr)