/*This file should only be included in CUDA cpp files*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "device/atomic.h"
#include "device/cuda.h"
#include "device/gpustd.h"
#include "myFilm.h"
#include <tiny-cuda-nn/common.h>

#include "nrrsparams.h"
#include "render/nrc/nrctrain.h"
#include "workqueue.h"
#include "render/adrrs-nrc/adntrain.h"
#include "workitem.h"

NAMESPACE_BEGIN(krr)

// position[3], incoming direction[2], roughness[1]
struct NRRSNetworkInput0 {
	Vector3f mPos; // normalized pos to [0, 1]^3
	Vector2f mDir; // normalized dir with 1-norm
#ifdef NRRS_USE_ROUGHNESS
	float mRoughness; // roughness
#endif
};

// loss.evaluate() will use float as target
// L[3], L^2[3]
struct NRRSNetworkOutputTraining0 {
	RGB mL;
	RGB mL2; // L^2 or Sigma
};

// make sure the size of NRRSNetworkInput0 is 16 __half aligned
struct NRRSNetworkInput1 {
	__half mL[3];
	__half mL2[3]; // L^2 or Sigma
	__half mThp[3];
#ifdef NRRS_USE_ROUGHNESS
	__half mRoughness;
#endif
#ifdef NRRS_USE_REF_MEAN
	__half mRefMean;
#endif
#ifdef NRRS_USE_POS_DIR
	__half mPos[3];
	__half mDir[2];
#else
	// paddings
	__half mPadding0[5];
#endif
};

struct NRRSNetworkInput1AID {
	Vector3f mPos;
	Vector2f mDir;
#ifdef NRRS_USE_ROUGHNESS
	float mRoughness;
#endif
#ifdef NRRS_USE_REF_MEAN
	float mRefMean;
#endif
	RGB mThp;
};

struct NRRSNetworkOutputTraining1 {
	float mRRS;
};

constexpr uint32_t NRRS_LL2NET_DIM_INPUT  = sizeof(NRRSNetworkInput0) / sizeof(float);
constexpr uint32_t NRRS_LL2NET_DIM_OUTPUT = sizeof(NRRSNetworkOutputTraining0) / sizeof(float);
constexpr uint32_t NRRS_LL2NET_DIM_OUTPUT_PADDED = 16;

constexpr uint32_t NRRS_LL2NET_INPUT_POS_OFFSET = offsetof(NRRSNetworkInput0, mPos) / sizeof(float);
constexpr uint32_t NRRS_LL2NET_INPUT_DIR_OFFSET = offsetof(NRRSNetworkInput0, mDir) / sizeof(float);
#ifdef NRRS_USE_ROUGHNESS
constexpr uint32_t NRRS_LL2NET_INPUT_ROUGHNESS_OFFSET =
	offsetof(NRRSNetworkInput0, mRoughness) / sizeof(float);
#endif

constexpr uint32_t NRRS_RRSNET_DIM_INPUT  = sizeof(NRRSNetworkInput1) / sizeof(__half);
constexpr uint32_t NRRS_RRSNET_DIM_OUTPUT = sizeof(NRRSNetworkOutputTraining1) / sizeof(float);
constexpr uint32_t NRRS_RRSNET_DIM_OUTPUT_PADDED = 16;
constexpr uint32_t NRRS_RRSNET_DIM_INPUT_PADDING =
	(NRRS_RRSNET_DIM_INPUT - offsetof(NRRSNetworkInput1, mPadding0[0]) / sizeof(__half));

constexpr uint32_t NRRS_RRSNET_DIM_INPUT_AID = sizeof(NRRSNetworkInput1AID) / sizeof(float);

template class NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0>;

__device__ inline float warp_roughness(const float roughness) { return 1 - expf(-roughness); }

__global__ void check_nan(const size_t nElements, const float *data);

__global__ void
nrrs_generate_inference_data_from_0to1(const size_t nElements, const NRRSInferenceQueue *inferQueue,
									   const float *net0Input, const __half *prediction_l,
									   const __half *prediction_l2, MyFilm *renderedImage,
									   const uint *pixelID, NRRSNetworkInput1 *net1Input);

__global__ void nrrs_generate_training_data_from_0to1(
	const size_t nElements, NRRSNetworkInput1 *net1input, const float *net0Input,
	const __half *predictionL, const __half *predictionL2, const float *refMeanForTraining,
	const float *thpForTraining, __half *ll2PtrForTraining, const bool trainSigma);

__global__ void nrrs_generate_training_data_from_0to1_aid(
	const size_t nElements, NRRSNetworkInput1AID *net1input, const float *net0Input,
	const __half *predictionL, const __half *predictionL2, const float *refMeanForTraining,
	const float *thpForTraining, __half *ll2PtrForTraining);

__global__ void nrrs_generate_inference_data(const size_t nElements,
											 const NRRSInferenceQueue *inferQueue, float *net0Input,
											 const AABB sceneAABB);

__global__ void nrrs_generate_inference_data_aid(const size_t nElements,
												 const NRRSInferenceQueue *inferQueue,
												 NRRSNetworkInput1AID *net1Input,
												 const AABB sceneAABB, MyFilm *renderedImage,
												 const uint *pixelID);

__global__ void nrrs_generate_training_data(
	const size_t nElements, NRRSPathNodesBuffer *pathState,
	NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0> *trainBuffer,
	NRRSNetworkOutputTraining1 *net1Output, const AABB sceneAABB, float *thpForTraining,
	float *errorForTraining, float *errorForAvgForTraining, float *refForTraining,
	float *errorPixel, MyFilm *renderedImage, NRRSPixelStateBuffer *pixelState, uint *rayPixelID,
	const bool ignoreZeroL, const int debugPixelID, float *numSamples, const uint errorImageScale,
	const uint frameSizeWidth, const float *numberSamplesThisPatch, const bool copyTheRRSNode,
	uint *pathStateNodeIdxAtomicBuffer, const bool ignoreSameTrainingDataForRRSNet);

__global__ void upload_denoise_error_buffer(const size_t nElements, const bool chooseError,
											const float *errorPerPixel, const RGB *weightedL,
											RGB *color);

__global__ void download_denoise_error_buffer(const size_t nElements, const bool chooseError,
											  float *errorPerPixel, RGB *weightedL,
											  const RGB *color, const bool errorMultiplySamples,
											  float *numberSamplesThisPatch,
											  const uint resolutionSize, float *errorFactorXX,
											  float *errorFactorXY);

__global__ void nrrs_update_weighted_L(const size_t nElements, NRRSPixelStateBuffer *psBuffer,
									   const RGB *weightedLCurrent, const uint *weightedLCurrentAcc,
									   RGB *weightedL, const float blendWeightForError);

__global__ void nrrs_calculate_error(const size_t nElements, float *error, MyFilm *renderedImage,
									 RGB *renderedImageDenoised, const bool useAccBuffer,
									 NRRSPixelStateBuffer *psBuffer, const RGB *weightedL,
									 const bool clamp, const bool errorMultiplySamples,
									 float *numberSamplesThisPatch, const uint resolutionSize,
									 float *errorFactorXX, float *errorFactorXY,
									 const bool denoisedOn);

__global__ void nrrs_calculate_sample_weight(
	const size_t nElements, float *sampleWeight, const uint *pixelIDArray,
	NRRSPixelStateBuffer *psBuffer,
	NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0> *trainBuffer);

__global__ void nrrs_UBS_calc_error(const size_t nElements, float *error, MyFilm *renderedImage,
									NRRSPixelStateBuffer *psBuffer, const int spp);

// gradient_1 += gradient_2
__global__ void calculate_gradient_sum(const uint32_t n_elements, __half *__restrict__ gradient_1,
									   const __half *__restrict__ gradient_2);

__global__ void gen_for_l2_evauate(const uint32_t n_elements, __half *__restrict__ nelLOutput,
								   float *__restrict__ target);

NAMESPACE_END(krr)