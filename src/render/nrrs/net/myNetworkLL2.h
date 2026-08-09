#pragma once
#include "networkcommon.h"
#include "twoHeadedNetwork.h"
#include "../myFilm.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

enum LL2TwoHeadType {
	// |                           + -- L
	// |  -- encoding -- net12 --- + -- L2
	AllShare = 0,
	// |                          + --- net2 -- L
	// |  -- encoding -- net1 --- + --- net2 -- L2
	PartShare = 1,
	// |                 + --- net12 -- L
	// |  -- encoding -- + --- net12 -- L2
	JustEncoding = 2,
	// |  -- encoding ---- net12 -- L
	// |  -- encoding ---- net12 -- L2
	AllSelf = 3,

	TH_Count = 4
};

class MyNetworkLL2 {
public:
	MyNetworkLL2(json &config);

	~MyNetworkLL2();

	void reset();

	// train
	float trainAndGenRRSNetInput(cudaStream_t stream, GPUMatrix_Float &input,
								 GPUMatrix_Float &output, const bool getLoss, const bool debugLoss,
								 const bool trainSigma, const bool onlyLL2, const float *thpPtr,
								 const float *refMeanPtr, float *input1Ptr, __half *LL2Ptr,
								 const bool net1AidNet2);

	float trainAndGenRRSNetInput_single(cudaStream_t stream, GPUMatrix_Float &input,
										GPUMatrix_Float &output, const bool getLoss,
										const bool debugLoss, const bool trainSigma,
										const bool onlyLL2, const float *thpPtr,
										const float *refMeanPtr, float *input1Ptr, __half *LL2Ptr,
										const bool net1AidNet2);
	float trainAndGenRRSNetInput_twoHead(cudaStream_t stream, GPUMatrix_Float &input,
										 GPUMatrix_Float &output, const bool getLoss,
										 const bool debugLoss, const bool trainSigma,
										 const bool onlyLL2, const float *thpPtr,
										 const float *refMeanPtr, float *input1Ptr, __half *LL2Ptr,
										 const bool net1AidNet2);

	void debugLossPrint(const cudaStream_t stream, const float trainSigma, const int nums,
						const int loss, const precision_t *output_l, const precision_t *output_l2,
						const precision_t *grad_l, const precision_t *grad_l2);

	// inference
	void inferenceAndGenRRSNetInput(cudaStream_t stream, const bool onlyLL2,
									const NRRSInferenceQueue *inferQueue, const int batchSizePad,
									MyFilm *renderedImage, const uint *pixelID,
									const AABB &sceneBoundingBox, NRRSNetworkInput1 *input1Ptr);
	void inferenceAndGenRRSNetInput_single(cudaStream_t stream, const bool onlyLL2,
										   const NRRSInferenceQueue *inferQueue,
										   const int batchSizePad, MyFilm *renderedImage,
										   const uint *pixelID, const AABB &sceneBoundingBox,
										   NRRSNetworkInput1 *input1Ptr);
	void inferenceAndGenRRSNetInput_twoHead(cudaStream_t stream, const bool onlyLL2,
											const NRRSInferenceQueue *inferQueue,
											const int batchSizePad, MyFilm *renderedImage,
											const uint *pixelID, const AABB &sceneBoundingBox,
											NRRSNetworkInput1 *input1Ptr);

	void updateLossHyperparams(const json &config);
	void resetTraining();
	void renderUI();

	void initBuffers(const int maxInferSize);

	float *getInferenceInputBufferPtr() { return mInferenceInputBuffer.data(); }
	precision_t *getInferenceOutputBufferPtr_L() { return mInferenceOutputBuffer.data(); }
	precision_t *getInferenceOutputBufferPtr_L2() { return mOutputPtr_L2; }

private:
	json mConfig;
	static constexpr int L2_OFFSET = 3;
	// two head net
	std::shared_ptr<TwoHeadedNetwork> mNetworkTwoHead{nullptr};

	// single net
	std::shared_ptr<NetworkWithInputEncoding> mNetwork{nullptr};
	std::shared_ptr<Optimizer> mOptimizer{nullptr};
	std::shared_ptr<Loss> mLoss{nullptr};
	std::shared_ptr<Trainer_LL2> mTrainer{nullptr};

	GPUMemory<float> mInferenceInputBuffer;
	GPUMemory<precision_t> mInferenceOutputBuffer;
	GPUMemory<precision_t> mInferenceOutputBuffer2; // for two head net
	precision_t *mOutputPtr_L2{nullptr};			// for inline functions

	LL2TwoHeadType mLL2TwoHeadType{LL2TwoHeadType::PartShare};
	bool mUseTwoHead{false};

	// temp
	constexpr static uint mTempGPUBufferSize = 128;
	float *mTempGPUBuffer{nullptr}; // cost 4
};

NAMESPACE_END(krr)