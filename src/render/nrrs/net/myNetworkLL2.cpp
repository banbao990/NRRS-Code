#include "myNetworkLL2.h"
#include "../train.h"

#include <tiny-cuda-nn/reduce_sum.h>
#include "twn_PartShare.h"
#include "twn_AllSelf.h"
#include "twn_JustEncoding.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

MyNetworkLL2::MyNetworkLL2(json &config) {
	// Auxiliary
	if (!mTempGPUBuffer) {
		cudaMalloc(&mTempGPUBuffer, sizeof(float) * mTempGPUBufferSize);
	}

	// Network
	mConfig = config;

	mUseTwoHead = config.value("ll2_use_two_head", false);
	int type	= config.value("ll2_two_head_type", 0);
	if (type <= LL2TwoHeadType::AllShare && type >= LL2TwoHeadType::TH_Count) {
		Log(Warning,
			"[MyNetworkLL2] LL2TwoHeadType is not valid, use PartShare instead! your setting = %d",
			type);
		type = 1; // PartShare
	}
	Log(Info, "[MyNetworkLL2] LL2TwoHeadType = %d", type);
	mLL2TwoHeadType = LL2TwoHeadType(type);

	reset();
}

MyNetworkLL2::~MyNetworkLL2() {
	if (mTempGPUBuffer) {
		cudaFree(mTempGPUBuffer);
	}
}

float MyNetworkLL2::trainAndGenRRSNetInput_single(cudaStream_t stream, GPUMatrix_Float &input,
												  GPUMatrix_Float &output, const bool getLoss,
												  const bool debugLoss, const bool trainSigma,
												  const bool onlyLL2, const float *thpPtr,
												  const float *refMeanPtr, float *input1Ptr,
												  __half *LL2Ptr, const bool net1AidNet2) {

	float ret = 0.0f; // loss

	// [STEP#1] train
	Context_Float ctx	   = mTrainer->training_step(stream, input, output);
	precision_t *netOutput = ctx->output.data();

	if (getLoss) {
		ret += mTrainer->loss(stream, *ctx);
	}

	// debug
	if (debugLoss || ret >= 1e4 || isnan(ret)) {
		int nums = ctx->output.n();

		const precision_t *output_l	 = netOutput;
		const precision_t *output_l2 = output_l + L2_OFFSET;
		const precision_t *grad_l	 = ctx->dL_doutput.data();
		const precision_t *grad_l2	 = grad_l + L2_OFFSET;

		debugLossPrint(stream, trainSigma, nums, ret, output_l, output_l2, grad_l, grad_l2);
	}

	if (onlyLL2) {
		return ret;
	}

	// [STEP#2] construct input for RRS network
	// the real size <= input.cols()

	const uint32_t num = input.cols();
	if (net1AidNet2) {
		LinearKernel(nrrs_generate_training_data_from_0to1_aid, stream, num,
					 (NRRSNetworkInput1AID *) input1Ptr, input.data(), netOutput,
					 netOutput + L2_OFFSET, refMeanPtr, thpPtr, LL2Ptr);
	} else {
		LinearKernel(nrrs_generate_training_data_from_0to1, stream, num,
					 (NRRSNetworkInput1 *) input1Ptr, input.data(), netOutput,
					 netOutput + L2_OFFSET, refMeanPtr, thpPtr, LL2Ptr, trainSigma);
	}

	return ret;
}

float MyNetworkLL2::trainAndGenRRSNetInput_twoHead(cudaStream_t stream, GPUMatrix_Float &input,
												   GPUMatrix_Float &output, const bool getLoss,
												   const bool debugLoss, const bool trainSigma,
												   const bool onlyLL2, const float *thpPtr,
												   const float *refMeanPtr, float *input1Ptr,
												   __half *LL2Ptr, const bool net1AidNet2) {

	float ret = 0.0f; // loss

	// [STEP#1] train
	precision_t *output1, *output2;
	ret += mNetworkTwoHead->training_step(stream, input, output, getLoss, output1, output2);

	// debug
	if (debugLoss || ret >= 1e4 || isnan(ret)) {
		Log(Info, "loss: %d\n", ret);
	}

	if (onlyLL2) {
		return ret;
	}

	// [STEP#2] construct input for RRS network
	// the real size <= input.cols()

	const uint32_t num = input.cols();
	if (net1AidNet2) {
		LinearKernel(nrrs_generate_training_data_from_0to1_aid, stream, num,
					 (NRRSNetworkInput1AID *) input1Ptr, input.data(), output1, output2, refMeanPtr,
					 thpPtr, LL2Ptr);
	} else {
		LinearKernel(nrrs_generate_training_data_from_0to1, stream, num,
					 (NRRSNetworkInput1 *) input1Ptr, input.data(), output1, output2, refMeanPtr,
					 thpPtr, LL2Ptr, trainSigma);
	}

	return ret;
}

void MyNetworkLL2::debugLossPrint(const cudaStream_t stream, const float trainSigma, const int nums,
								  const int loss, const precision_t *output_l,
								  const precision_t *output_l2, const precision_t *grad_l,
								  const precision_t *grad_l2) {

	// check the item each
	constexpr uint sumBufferSize	  = 4;
	float sumBufferCPU[sumBufferSize] = {};
	{
		float *sumBuffer = mTempGPUBuffer;
		// reset
		GPUCall(
			[=] KRR_DEVICE() mutable {
				for (int i = 0; i < sumBufferSize; i++) {
					sumBuffer[i] = 0.0f;
				}
			},
			stream);

		// sum
		GPUParallelFor(
			nums,
			[=] KRR_DEVICE(int i) mutable {
				const uint32_t offset	  = i * NRRS_LL2NET_DIM_OUTPUT_PADDED; // all padding to 16
				const precision_t *out[4] = {output_l, output_l2, grad_l, grad_l2};
				for (int i = 0; i < 4; ++i) {
					const precision_t *s = out[i] + offset;
					float l				 = float(s[0] + s[1] + s[2]);
					atomicAdd(sumBuffer + i, l);
				}
			},
			stream);

		cudaMemcpy(sumBufferCPU, sumBuffer, sizeof(float) * sumBufferSize, cudaMemcpyDeviceToHost);
		for (int i = 0; i < sumBufferSize; ++i) {
			sumBufferCPU[i] = sumBufferCPU[i] / nums;
		}
	}

	Log(Info, "loss: %f, output(L/L2): %f/%f, gradient(L/L2): %f/%f", loss, sumBufferCPU[0],
		sumBufferCPU[1], sumBufferCPU[2], sumBufferCPU[3]);
}

float MyNetworkLL2::trainAndGenRRSNetInput(cudaStream_t stream, GPUMatrix_Float &input,
										   GPUMatrix_Float &output, const bool getLoss,
										   const bool debugLoss, const bool trainSigma,
										   const bool onlyLL2, const float *thpPtr,
										   const float *refMeanPtr, float *input1Ptr,
										   __half *LL2Ptr, const bool net1AidNet2) {
	if (mUseTwoHead) {
		return trainAndGenRRSNetInput_twoHead(stream, input, output, getLoss, debugLoss, trainSigma,
											  onlyLL2, thpPtr, refMeanPtr, input1Ptr, LL2Ptr,
											  net1AidNet2);
	} else {
		return trainAndGenRRSNetInput_single(stream, input, output, getLoss, debugLoss, trainSigma,
											 onlyLL2, thpPtr, refMeanPtr, input1Ptr, LL2Ptr,
											 net1AidNet2);
	}
}

void MyNetworkLL2::inferenceAndGenRRSNetInput_single(cudaStream_t stream, const bool onlyLL2,
													 const NRRSInferenceQueue *inferQueue,
													 const int batchSizePad, MyFilm *renderedImage,
													 const uint *pixelID,
													 const AABB &sceneBoundingBox,
													 NRRSNetworkInput1 *input1Ptr) {

	// [STEP#1] LL2 inference
	float *inputPtr0		= mInferenceInputBuffer.data();
	precision_t *outputPtr0 = mInferenceOutputBuffer.data();

	const GPUMatrix_Float inferenceInputs(inputPtr0, NRRS_LL2NET_DIM_INPUT, batchSizePad);
	// inference_mix's output should padded
	GPUMatrix_Half inferenceOutputs(outputPtr0, NRRS_LL2NET_DIM_OUTPUT_PADDED, batchSizePad);
	// [IM] here we use __half, however during training, we use float for LL2Net

	mNetwork->inference_mixed_precision(stream, inferenceInputs, inferenceOutputs);

	if (onlyLL2) {
		return;
	}

	// [STEP#2] construct input for RRS network
	LinearKernel(nrrs_generate_inference_data_from_0to1, stream, batchSizePad, inferQueue,
				 inputPtr0, outputPtr0, outputPtr0 + L2_OFFSET, renderedImage, pixelID, input1Ptr);
}

void MyNetworkLL2::inferenceAndGenRRSNetInput(cudaStream_t stream, const bool onlyLL2,
											  const NRRSInferenceQueue *inferQueue,
											  const int batchSizePad, MyFilm *renderedImage,
											  const uint *pixelID, const AABB &sceneBoundingBox,
											  NRRSNetworkInput1 *input1Ptr) {

	if (mUseTwoHead) {
		inferenceAndGenRRSNetInput_twoHead(stream, onlyLL2, inferQueue, batchSizePad, renderedImage,
										   pixelID, sceneBoundingBox, input1Ptr);
	} else {
		inferenceAndGenRRSNetInput_single(stream, onlyLL2, inferQueue, batchSizePad, renderedImage,
										  pixelID, sceneBoundingBox, input1Ptr);
	}
}

void MyNetworkLL2::inferenceAndGenRRSNetInput_twoHead(cudaStream_t stream, const bool onlyLL2,
													  const NRRSInferenceQueue *inferQueue,
													  const int batchSizePad, MyFilm *renderedImage,
													  const uint *pixelID,
													  const AABB &sceneBoundingBox,
													  NRRSNetworkInput1 *input1Ptr) {

	// [STEP#1] LL2 inference
	float *inputPtr0		= mInferenceInputBuffer.data();
	precision_t *outputPtr0 = mInferenceOutputBuffer.data();
	precision_t *outputPtr1 = mInferenceOutputBuffer2.data();

	const GPUMatrix_Float inferenceInput(inputPtr0, NRRS_LL2NET_DIM_INPUT, batchSizePad);
	// inference_mix's output should padded
	GPUMatrix_Half inferenceOutput1(outputPtr0, NRRS_LL2NET_DIM_OUTPUT_PADDED, batchSizePad);
	// [IM] here we use __half, however during training, we use float for LL2Net
	GPUMatrix_Half inferenceOutput2(outputPtr1, NRRS_LL2NET_DIM_OUTPUT_PADDED, batchSizePad);

	mNetworkTwoHead->inference(stream, inferenceInput, inferenceOutput1, inferenceOutput2);

	if (onlyLL2) {
		return;
	}

	// [STEP#2] construct input for RRS network
	LinearKernel(nrrs_generate_inference_data_from_0to1, stream, batchSizePad, inferQueue,
				 inputPtr0, outputPtr0, outputPtr1, renderedImage, pixelID, input1Ptr);
}

void MyNetworkLL2::updateLossHyperparams(const json &config) {
	if (mUseTwoHead) {
		mNetworkTwoHead->updateLossHyperparams(config);
	} else {
		mLoss->update_hyperparams(config);
	}
}

void MyNetworkLL2::resetTraining() { mTrainer->initialize_params(); }

void MyNetworkLL2::renderUI() {
	if (mUseTwoHead) {
		ui::TextColored(ImVec4(1.0, 0, 0, 1.0), "[LL2Net] Use Two-Head Network");
	} else {
		ui::TextColored(ImVec4(1.0, 0, 0, 1.0), "[LL2Net] Use Single-Head Network");
	}
}

void MyNetworkLL2::initBuffers(const int maxInferSize) {
	mInferenceInputBuffer  = GPUMemory<float>(NRRS_LL2NET_DIM_INPUT * maxInferSize);
	mInferenceOutputBuffer = GPUMemory<precision_t>(NRRS_LL2NET_DIM_OUTPUT_PADDED * maxInferSize);
	if (mUseTwoHead) {
		mNetworkTwoHead->set_max_training_batch_size(NRRS_TRAIN_BATCH_SIZE);
		mNetworkTwoHead->set_max_inference_batch_size(maxInferSize);
		mInferenceOutputBuffer2 =
			GPUMemory<precision_t>(NRRS_LL2NET_DIM_OUTPUT_PADDED * maxInferSize);
		mOutputPtr_L2 = mInferenceOutputBuffer2.data();
	} else {
		mOutputPtr_L2 = mInferenceOutputBuffer.data() + L2_OFFSET;
	}
}

void MyNetworkLL2::reset() {
	if (mUseTwoHead) {
		// keep all buffers
		if (mNetworkTwoHead) {
			mNetworkTwoHead->reset();
		} else {
			switch (mLL2TwoHeadType) {
				case LL2TwoHeadType::PartShare:
					mNetworkTwoHead = std::make_shared<TwoHeadedNetwork_PartShare>(
						mConfig, NRRS_LL2NET_DIM_INPUT, NRRS_LL2NET_DIM_OUTPUT);
					break;
				case LL2TwoHeadType::JustEncoding:
					mNetworkTwoHead = std::make_shared<TwoHeadedNetwork_JustEncoding>(
						mConfig, NRRS_LL2NET_DIM_INPUT, NRRS_LL2NET_DIM_OUTPUT);
					break;
				case LL2TwoHeadType::AllSelf:
					mNetworkTwoHead = std::make_shared<TwoHeadedNetwork_AllSelf>(
						mConfig, NRRS_LL2NET_DIM_INPUT, NRRS_LL2NET_DIM_OUTPUT);
					break;
				case LL2TwoHeadType::AllShare:
				case LL2TwoHeadType::TH_Count:
				default:
					Log(Fatal, "[%s::%d] LL2TwoHeadType is not valid, your setting = %d",
						mLL2TwoHeadType, __FILE__, __LINE__);
					break;
			}
			if (!mNetworkTwoHead) {
				Log(Fatal, "[%s::%d] Failed to create `TwoHeadedNetwork` instance.", __FILE__,
					__LINE__);
			}
			mNetworkTwoHead->print_info();
		}
	} else {
		const json &encoding_config	 = mConfig["encoding"];
		const json &optimizer_config = mConfig["optimizer"];
		json &network_config		 = mConfig["network"];
		const json &loss_config		 = mConfig["loss"];

		const uint32_t n_neurons	= mConfig["n_neurons"];
		network_config["n_neurons"] = n_neurons;

		mOptimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
		mLoss.reset(tcnn::create_loss<precision_t>(loss_config));
		mNetwork = std::make_shared<NetworkWithInputEncoding>(
			NRRS_LL2NET_DIM_INPUT, NRRS_LL2NET_DIM_OUTPUT, encoding_config, network_config);

		mTrainer = std::make_shared<Trainer_LL2>(mNetwork, mOptimizer, mLoss, KRR_DEFAULT_RND_SEED);

		const auto paddedOutputWidth = mNetwork->padded_output_width();
		Log(Info, "Network has a padded output width of %d", paddedOutputWidth);

		// params
		int net_params		 = mNetwork->n_params();
		int encoding_params	 = mNetwork->encoding()->n_params();
		int trainer_params	 = mTrainer->n_params();
		int optimizer_params = mOptimizer->n_weights();
		Log(Info,
			"Network has %d (%d + %d) parameters. Trainer has %d parameters. Optimizer has %d "
			"parameters.",
			net_params, encoding_params, (net_params - encoding_params), trainer_params,
			optimizer_params);
	}
}

NAMESPACE_END(krr)
