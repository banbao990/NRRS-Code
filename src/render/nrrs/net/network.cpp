#include "network.h"
#include "../nrrsparams.h"
#include "../train.h"
#include "myNetworkLL2.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

std::string getTimestamp() {
	time_t now			  = time(0);
	tm *ltm				  = localtime(&now);
	std::string timestamp = std::to_string(1900 + ltm->tm_year) + "-" +
							std::to_string(1 + ltm->tm_mon) + "-" + std::to_string(ltm->tm_mday) +
							"-" + std::to_string(ltm->tm_hour) + "-" + std::to_string(ltm->tm_min) +
							"-" + std::to_string(ltm->tm_sec);
	return timestamp;
}

std::chrono::steady_clock::time_point currentTime() {
	return std::chrono::high_resolution_clock::now();
}

float deltaTime(std::chrono::steady_clock::time_point t1,
				std::chrono::steady_clock::time_point t2) {
	return std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
}

NRRSNetwork::NRRSNetwork(json &config) {
	mConfig = config;

	mNet1AID = config.value("net1_aid_net2", false);
	Log(Info, "[NRRSNetwork] Net1AID: %s", mNet1AID ? "true" : "false");

	cudaMalloc(&mLossSumErrorGPUPtr, sizeof(float));
}

NRRSNetwork::~NRRSNetwork() {
	if (mLossSumErrorGPUPtr) cudaFree(mLossSumErrorGPUPtr);
	if (mThpForTraining) cudaFree(mThpForTraining);
	if (mErrorForTraining) cudaFree(mErrorForTraining);
	if (mErrorForAvgForTraining) cudaFree(mErrorForAvgForTraining);
	if (mErrorPerPixel) cudaFree(mErrorPerPixel);
	if (mPixelDebugBuffer) cudaFree(mPixelDebugBuffer);
	if (mRefMeanForTraining) cudaFree(mRefMeanForTraining);
	if (mSampleWeightForTraining) cudaFree(mSampleWeightForTraining);
	if (mPixelIDForTraining) cudaFree(mPixelIDForTraining);
	if (mNumberSamplesForTraining) cudaFree(mNumberSamplesForTraining);
}

void NRRSNetwork::renderUI() {
	ui::PushID("NRRSNetwork");
	if (ui::TreeNode("LL2Network Infos")) {
		mNetworkLL2->renderUI();
		ui::TreePop();
	}

	ui::Checkbox("Debug Loss", &mDebugLoss);

	// loss
	if (ui::TreeNodeEx("Loss", ImGuiTreeNodeFlags_DefaultOpen)) {
		ui::PushID("Loss");
		bool lossChanged = false;
		lossChanged |= ui::Checkbox("clamp on", &mLossClampOn);
		lossChanged |= ui::SliderFloat("clamp max", &mLossClampMax, 0.0f, 1000.0f);
		lossChanged |= ui::Checkbox("Train Sigma(E[x^2] if false)", &mTrainSigmaOrX2);
		static const char *sLossStep[] = {"step 0", "step 1", "step 2", "step 3"};
		lossChanged |= ui::Combo("step", &mLossStep, sLossStep, std::size(sLossStep));

		static const char *sLossNames[ShowLossType::Count] = {
			"None",
			"L+L^2+rrs",
			"L+L^2",
			"rrs",
			"grad(all)",
			"grad(rrs, avg E)",
			"grad(rrs, min E)",
			"grad(rrs, min rrs)",
		};
		lossChanged |= ui::Combo("show loss index(just show, won't effect result)",
								 (int *) &mLossShowLossIndex, sLossNames, std::size(sLossNames));

		ui::Checkbox("[Var] Clamp pixel error", &mClampPixelError);
		lossChanged |=
			ui::Checkbox("[Var] Pixel error multiply samples", &mPixelErrorMultiplySamples);

		if (ui::TreeNode("Params")) {
			static float gammaMaxLogScale = 0.0f;

			ui::SliderFloat("gamma ax in log scale", &gammaMaxLogScale, -3.0f, 4.0f, "%.2f");
			float gammaMax = pow(10.0f, gammaMaxLogScale);
			lossChanged |= ui::SliderFloat("gamma1", &mGamma1, 0.0001f, gammaMax, "%.4f");
			lossChanged |= ui::SliderFloat("gamma2", &mGamma2, 0.0001f, gammaMax, "%.4f");
			lossChanged |= ui::SliderFloat("gamma3", &mGamma3, 0.0001f, gammaMax, "%.4f");
			lossChanged |= ui::SliderFloat("gamma4", &mGamma4, 0.0001f, gammaMax, "%.4f");
			ui::Text("g1 = %.4f, g2 = %.4f, g3 = %.4f, g4 = %.4f", mGamma1, mGamma2, mGamma3,
					 mGamma4);

			ui::TreePop();
		}

		lossChanged |= ui::Checkbox("[RRS Net] Relief Error", &mReliefError);

		lossChanged |= ui::SliderInt("A Debug Int Pass to Loss", &mShowNetworkDebugInt, 0, 20);

		if (lossChanged) {
			mLoss->update_hyperparams({
				{"clamp_on", mLossClampOn},
				{"clamp_max", mLossClampMax},
				{"train_sigma", mTrainSigmaOrX2},
				{"step", mLossStep},
				{"show_loss_index", (int) mLossShowLossIndex},
				{"gamma1", mGamma1},
				{"gamma2", mGamma2},
				{"gamma3", mGamma3},
				{"gamma4", mGamma4},
				{"pixel_error_multiply_samples", mPixelErrorMultiplySamples},
				{"relief_error", mReliefError},
				{"debug_int", mShowNetworkDebugInt},
			});
		}

		ui::PopID();
		ui::TreePop();
	}

	// learning rate
	static float lrRRSNet = mOptimizer->learning_rate();
	if (ui::DragFloat("RRS Net: Learning rate", &lrRRSNet, 1e-6f, 0, 1e-1, "%.6f")) {
		mOptimizer->set_learning_rate(lrRRSNet);
	}

	if (ui::TreeNodeEx("Pixel Debug Buffer Params", ImGuiTreeNodeFlags_DefaultOpen)) {
		static float sMaxLogScale = 0.0f;
		ui::SliderFloat("max log scale", &sMaxLogScale, -2.0f, 4.0f, "%.2f");
		float sMax = pow(10.0f, sMaxLogScale);
		bool vc	   = false;
		vc |= ui::SliderFloat("[RRS-Net] debug float", &mDebugFloat, 0.0f, sMax, "%.4f");
		if (vc) {
			mLoss->update_hyperparams({{"debug_float", mDebugFloat}});
		}

		ui::Checkbox("Reset pixel debug buffer each frame", &mResetPixelDebugBufferEachFrame);
		bool resetDebugPixelBuffer = ui::Button("Reset debug pixel buffer");
		resetDebugPixelBuffer |= mResetPixelDebugBufferEachFrame;
		if (resetDebugPixelBuffer) {
			cudaMemsetAsync(mPixelDebugBuffer, 0, sizeof(uint) * mResolutionSize,
							gpContext->cudaStream);
			mPixelDebugCount = 0;
		}
		ui::TreePop();
	}

	// ui end
	ui::PopID();
}

// [TODO] very slow
[[deprecated("maybe error")]]
void NRRSNetwork::loadWeights(const std::string &filePath) {
	Log(Fatal, "Not implemented yet");

	auto t1 = currentTime();

	auto weights = File::loadJSON(filePath);

	auto t2 = currentTime();

	// mTrainer->deserialize(weights);

	auto t3 = currentTime();

	Log(Info, "Weights loaded from %s, load time: %f s, deserialize time: %f s", filePath.c_str(),
		deltaTime(t1, t2), deltaTime(t2, t3));
}

// [TODO] very slow
[[deprecated("maybe error")]]
void NRRSNetwork::saveWeights(const std::string &filePath) {
	Log(Fatal, "Not implemented yet");

	auto t1 = currentTime();

	// auto weights  = mTrainer->serialize(true);
	auto fileName = filePath + "/" + getTimestamp() + ".json";

	auto t2 = currentTime();

	// File::saveJSON(fileName, weights);

	auto t3 = currentTime();

	Log(Info, "Weights saved to %s, deserialize time: %f s, save time: %f s", fileName.c_str(),
		deltaTime(t1, t2), deltaTime(t2, t3));
}

float NRRSNetwork::train(GPUMatrix_Float &input, GPUMatrix_Float &output,
						 const int offsetOfTrainingData, const bool onlyLL2) {
	const cudaStream_t stream = gpContext->cudaStream;
	float ret				  = 0.0f;

	{ // ll2
		bool getLoss = (mLossShowLossIndex == ShowLossType::LL2_Loss) ||
					   (mLossShowLossIndex == ShowLossType::All);

		const float *thpPtr		= mThpForTraining + offsetOfTrainingData * 3; // 3 floats
		const float *refMeanPtr = mRefMeanForTraining + offsetOfTrainingData; // 1 float

		// offset
		const int input1ElementSize = mNet1AID ? (sizeof(NRRSNetworkInput1AID) / sizeof(float))
											   : (sizeof(NRRSNetworkInput1) / sizeof(float));
		float *input1Ptr = mTrainingInputBuffer1.data() + offsetOfTrainingData * input1ElementSize;

		// TODO: when mNet1AID is true; we can construct the input of RRSNet in
		// nrrs_generate_training_data(); inference is the same

		// train & construct input for RRS network
		ret += mNetworkLL2->trainAndGenRRSNetInput(stream, input, output, getLoss, mDebugLoss,
												   mTrainSigmaOrX2, onlyLL2, thpPtr, refMeanPtr,
												   input1Ptr, mLL2Ptr, mNet1AID);
	}

	// static uint warmup = 0;
	// if (warmup < 500) {
	//	warmup++;
	//	return ret;
	// }

	if (onlyLL2) {
		return ret;
	}

	{ // rrs
		bool getLoss = (mLossShowLossIndex != ShowLossType::LL2_Loss) &&
					   (mLossShowLossIndex != ShowLossType::None);
		float rrsLoss = 0;

		const uint32_t num = input.cols();
		// in fact, during training, the output buffer is not used
		float *output1Ptr = (float *) (mTrainingOutputBuffer1.data() + offsetOfTrainingData);
		GPUMatrix_Float outputForNet1(output1Ptr, NRRS_RRSNET_DIM_OUTPUT, num);

		if (mNet1AID) {
			float *input1Ptr =
				mTrainingInputBuffer1.data() +
				offsetOfTrainingData * (sizeof(NRRSNetworkInput1AID) / sizeof(float));
			GPUMatrix_Float inputForNet1(input1Ptr, NRRS_RRSNET_DIM_INPUT_AID, num);
			auto ctx = mTrainerRRSAID->training_step(stream, inputForNet1, outputForNet1);

			if (getLoss) {
				rrsLoss = mTrainerRRSAID->loss(stream, *ctx);
				ret += rrsLoss;
			}

			// debug
			if (mDebugLoss || rrsLoss >= 1e4 || isnan(rrsLoss)) {
				// calculate gradient
				float gradient =
					reduce_sum(ctx->dL_doutput.data(), ctx->dL_doutput.n_elements(), stream);
				// only 1 in 16 elements are used
				float rrs = reduce_sum(ctx->output.data(), ctx->output.n_elements(), stream);
				Log(Info, "loss: %f, gradient: %f, rrs(avg): %f", rrsLoss, gradient,
					rrs / ctx->output.n_elements() * 16);
			}

		} else {
			precision_t *input1Ptr = (precision_t *) (mTrainingInputBuffer1.data() +
													  (sizeof(NRRSNetworkInput1) / sizeof(float)) *
														  offsetOfTrainingData);
			GPUMatrix_Half inputForNet1(input1Ptr, NRRS_RRSNET_DIM_INPUT, num);

			// LinearKernel(check_nan, stream, num * NRRS_RRSNET_DIM_INPUT, inputForNet1.data());
			// LinearKernel(check_nan, stream, num * NRRS_RRSNET_DIM_OUTPUT, outputForNet1.data());

			auto ctx = mTrainerRRS->training_step(stream, inputForNet1, outputForNet1);

			if (getLoss) {
				rrsLoss = mTrainerRRS->loss(stream, *ctx);
				ret += rrsLoss;
			}

			// debug
			if (mDebugLoss || rrsLoss >= 1e4 || isnan(rrsLoss)) {
				// calculate gradient
				float gradient =
					reduce_sum(ctx->dL_doutput.data(), ctx->dL_doutput.n_elements(), stream);
				// only 1 in 16 elements are used
				float rrs = reduce_sum(ctx->output.data(), ctx->output.n_elements(), stream);
				Log(Info, "loss: %f, gradient: %f, rrs(avg): %f", rrsLoss, gradient,
					rrs / ctx->output.n_elements() * 16);
			}
		}
	}

	return ret;
}

void NRRSNetwork::prepareInferenceData(const int inferenceQueueSize,
									   const NRRSInferenceQueue *inferQueue, const AABB &sceneAABB,
									   bool onlyRRSNet, MyFilm *renderedImage,
									   const uint *pixelID) {
	const cudaStream_t stream = gpContext->cudaStream;

	if (mNet1AID && onlyRRSNet) {
		NRRSNetworkInput1AID *net1Input = (NRRSNetworkInput1AID *) getInferenceInputBufferPtr1();
		LinearKernel(nrrs_generate_inference_data_aid, stream, inferenceQueueSize, inferQueue,
					 net1Input, sceneAABB, renderedImage, pixelID);
	} else {
		float *net0Input = getInferenceInputBufferPtr0();
		LinearKernel(nrrs_generate_inference_data, stream, inferenceQueueSize, inferQueue,
					 net0Input, sceneAABB);
	}
}

void NRRSNetwork::inference(const int inferenceQueueSize, const bool onlyLL2Net,
							const NRRSInferenceQueue *inferQueue, MyFilm *renderedImage,
							const uint *pixelID) {

	const cudaStream_t stream = gpContext->cudaStream;

	const int batchSizePad = next_multiple(inferenceQueueSize, (int) BATCH_SIZE_GRANULARITY);

	if (!mNet1AID || onlyLL2Net) {
		// LL2 inference
		mNetworkLL2->inferenceAndGenRRSNetInput(
			stream, onlyLL2Net, inferQueue, batchSizePad, renderedImage, pixelID,
			mScene->getBoundingBox(), (NRRSNetworkInput1 *) mInferenceInputBuffer1.data());
	}

	if (onlyLL2Net) {
		return;
	}

	{ // RRS inference
		precision_t *inputPtr  = mInferenceInputBuffer1.data();
		precision_t *outputPtr = mInferenceOutputBuffer1.data();

		// inference_mix's output should padded
		GPUMatrix_Half inferenceOutputs(outputPtr, NRRS_RRSNET_DIM_OUTPUT_PADDED, batchSizePad);

		if (mNet1AID) {
			GPUMatrix_Float inferenceInputs((float *) inputPtr, NRRS_RRSNET_DIM_INPUT_AID,
											batchSizePad);
			mNetworkAID->inference_mixed_precision(stream, inferenceInputs, inferenceOutputs);
		} else {
			GPUMatrix_Half inferenceInputs(inputPtr, NRRS_RRSNET_DIM_INPUT, batchSizePad);
			mNetworkRRS->inference_mixed_precision(stream, inferenceInputs, inferenceOutputs);
		}
	}
}

void NRRSNetwork::resize(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples) {
	const int resolutionSize = frameSize[0] * frameSize[1];
	if (!mNetIsInitialized) {
		// initialize the network
		mNetIsInitialized = true;
		reset(frameSize, maxQueueSize, maxTrainingSamples);
		return;
	}

	// [TODO] now no check for the size of the input buffer
	if (resolutionSize) {
		mResolutionSize = resolutionSize;
	}
	if (maxQueueSize) {
		mMaxQueueSize = maxQueueSize;
	}
	if (maxTrainingSamples) {
		mMaxTrainingSamples = maxTrainingSamples;
	}

	if (resolutionSize && maxQueueSize && maxTrainingSamples) {
		const auto &maxQS = maxQueueSize;

		const uint inferenceInput1Size = mNet1AID
											 ? (sizeof(NRRSNetworkInput1AID) / sizeof(precision_t))
											 : (sizeof(NRRSNetworkInput1) / sizeof(precision_t));

		mInferenceInputBuffer1	= GPUMemory<precision_t>(inferenceInput1Size * maxQS);
		mInferenceOutputBuffer1 = GPUMemory<precision_t>(NRRS_RRSNET_DIM_OUTPUT_PADDED * maxQS);

		mNetworkLL2->initBuffers(maxQS);

		// max size = BATCH_SIZE

		const uint trainInput1Size = mNet1AID ? (sizeof(NRRSNetworkInput1AID) / sizeof(float))
											  : (sizeof(NRRSNetworkInput1) / sizeof(float));

		mTrainingInputBuffer1  = GPUMemory<float>(maxTrainingSamples * trainInput1Size);
		mTrainingOutputBuffer1 = GPUMemory<NRRSNetworkOutputTraining1>(maxTrainingSamples);

		if (mLL2Ptr) {
			cudaFree(mLL2Ptr);
		}
		cudaMalloc(&mLL2Ptr, sizeof(precision_t) * 6 * NRRS_TRAIN_BATCH_SIZE);

		if (mErrorPerPixel) {
			cudaFree(mErrorPerPixel);
		}
		// record error1 after denoise [for min loss]
		// record error2 after error1 / sqrt(?) [for avg loss]
		cudaMalloc(&mErrorPerPixel, sizeof(float) * resolutionSize * 2);

		if (mPixelDebugBuffer) {
			cudaFree(mPixelDebugBuffer);
		}
		cudaMalloc(&mPixelDebugBuffer, sizeof(uint32_t) * resolutionSize);

#define BB_MALLOC(p, s, type)                                                                      \
	if (p) {                                                                                       \
		cudaFree(p);                                                                               \
	}                                                                                              \
	cudaMalloc(&p, (maxTrainingSamples * sizeof(type) * s))

		BB_MALLOC(mThpForTraining, 3, float);
		BB_MALLOC(mErrorForTraining, 1, float);
		BB_MALLOC(mErrorForAvgForTraining, 1, float);
		BB_MALLOC(mRefMeanForTraining, 1, float);
		BB_MALLOC(mSampleWeightForTraining, 1, float);
		BB_MALLOC(mPixelIDForTraining, 1, uint);
		BB_MALLOC(mNumberSamplesForTraining, 1, float);

#undef BB_MALLOC
		mLoss->update_hyperparams({
			{"error_per_pixel", (uint64_t) mErrorPerPixel},
			{"pixel_debug_buffer", (uint64_t) mPixelDebugBuffer},
			{"thp", (uint64_t) mThpForTraining},
			{"error", (uint64_t) mErrorForTraining},
			{"error_for_avg", (uint64_t) mErrorForAvgForTraining},
			{"ref_mean", (uint64_t) mRefMeanForTraining},
			{"sample_weight", (uint64_t) mSampleWeightForTraining},
			{"ll2", (uint64_t) mLL2Ptr},
			{"pixel_id", (uint64_t) mPixelIDForTraining},
			{"number_samples", (uint64_t) mNumberSamplesForTraining},
			{"frame_size_width", (uint32_t) frameSize[0]},
		});
	} else {
		if (maxQueueSize || maxTrainingSamples) {
			Log(Fatal, "(maxQueueSize, maxTrainingSamples) should be both 0 or all non-zero");
		}
	}
}

void NRRSNetwork::reset(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples) {
	const int resolutionSize  = frameSize[0] * frameSize[1];
	const cudaStream_t stream = gpContext->cudaStream;
	using namespace tcnn;
	tcnn::free_gpu_memory_arena(stream);

	if (mFirstLoadConfig) {
		mLossClampOn	   = mConfig.value("clamp_on", mLossClampOn);
		mLossClampMax	   = mConfig.value("clamp_max", mLossClampMax);
		mTrainSigmaOrX2	   = mConfig.value("train_sigma", true);
		mLossShowLossIndex = mConfig.value("show_loss_index", mLossShowLossIndex);
		mErrorImageScale   = mConfig.value("error_image_scale", 1u);
		mReliefError	   = mConfig.value("relief_error", false);
	}

	{ // level 0
		Log(Info, "Resetting NRRS network [level0]");

		auto &config = mConfig["level0"];

		config["ll2_use_two_head"]	= mConfig["ll2_use_two_head"];
		config["ll2_two_head_type"] = mConfig["ll2_two_head_type"];

		if (mNetworkLL2) {
			// keep all the buffers
			mNetworkLL2->reset();
		} else {
			mNetworkLL2 = std::make_shared<MyNetworkLL2>(config);
		}

		mNetworkLL2->updateLossHyperparams({
			{"clamp_on", mLossClampOn},
			{"clamp_max", mLossClampMax},
			{"train_sigma", mTrainSigmaOrX2},
			{"step", (uint32_t) mLossStep},
		});
	}

	{ // level 1
		Log(Info, "Resetting NRRS network [level1]");

		auto &config = mConfig["level1"];

		// [##network##]
		json &network_config = config["network"];

		network_config["n_output_dims"] = NRRS_RRSNET_DIM_OUTPUT;

		auto &net_aid = mNetworkAID;
		auto &net	  = mNetworkRRS;

		if (mNet1AID) {
			// network input dim is decided by encoding
			auto &encoding_aid = mConfig["level1_aid"]["encoding"];
			net_aid			   = std::make_shared<NetworkWithInputEncoding>(
				   NRRS_RRSNET_DIM_INPUT_AID, NRRS_RRSNET_DIM_OUTPUT, encoding_aid, network_config);

			Log(Info, "Network has a padded input width of %d", net_aid->input_width());
			Log(Info, "Network has a padded output width of %d", net_aid->padded_output_width());

		} else {
			network_config["n_input_dims"] = NRRS_RRSNET_DIM_INPUT;

			net.reset(create_network<precision_t>(network_config));

			Log(Info, "Network has a padded output width of %d", net->padded_output_width());
		}

		// [##loss##]]
		json &loss_config = config["loss"];
		auto &loss		  = mLoss;
		loss.reset(tcnn::create_loss<precision_t>(loss_config));

		if (mFirstLoadConfig) {
			// only update if it's the first time loading the config [for debug]
			mLossStep				   = loss_config.value("step", mLossStep);
			mClampPixelError		   = loss_config.value("clamp_pixel_error", true);
			mPixelErrorMultiplySamples = loss_config.value("pixel_error_multiply_samples", false);

			mGamma1 = loss_config.value("gamma1", 1.0f);
			mGamma2 = loss_config.value("gamma2", 1.0f);
			mGamma3 = loss_config.value("gamma3", 1.0f);
			mGamma4 = loss_config.value("gamma4", 1.0f);
			// mOnlyTrainL				   = loss_config.value("train_L2", mOnlyTrainL);
		}

		// mLossShowLossIndex = 0;
		// mLossStep		   = 1;

		loss->update_hyperparams({
			{"pixels", (uint32_t) resolutionSize},
			{"clamp_on", mLossClampOn},
			{"clamp_max", mLossClampMax},
			{"train_sigma", mTrainSigmaOrX2},
			{"step", (uint32_t) mLossStep},
			{"show_loss_index", (uint32_t) mLossShowLossIndex},
			{"error_sum_ptr", (uint64_t) mLossSumErrorGPUPtr},
			{"thp", (uint64_t) mThpForTraining},
			{"error", (uint64_t) mErrorForTraining},
			{"error_for_avg", (uint64_t) mErrorForAvgForTraining},
			{"ref_mean", (uint64_t) mRefMeanForTraining},
			{"sample_weight", (uint64_t) mSampleWeightForTraining},
			{"pixel_id", (uint64_t) mPixelIDForTraining},
			{"number_samples", (uint64_t) mNumberSamplesForTraining},
			{"error_per_pixel", (uint64_t) mErrorPerPixel},
			{"pixel_debug_buffer", (uint64_t) mPixelDebugBuffer},
			{"pixel_error_multiply_samples", mPixelErrorMultiplySamples},
			{"ll2", (uint64_t) mLL2Ptr},
			{"gamma1", mGamma1},
			{"gamma2", mGamma2},
			{"gamma3", mGamma3},
			{"gamma4", mGamma4},
			{"error_image_scale", (uint32_t) mErrorImageScale},
			{"relief_error", mReliefError},
		});

		// [##optimizer##]
		json &optimizer_config = config["optimizer"];
		auto &opti			   = mOptimizer;
		opti.reset(tcnn::create_optimizer<precision_t>(optimizer_config));

		// [##trainer##]
		if (mNet1AID) {
			mTrainerRRSAID =
				std::make_shared<Trainer_RRS_AID>(net_aid, opti, loss, KRR_DEFAULT_RND_SEED);
		} else {
			mTrainerRRS = std::make_shared<Trainer_RRS>(net, opti, loss, KRR_DEFAULT_RND_SEED);
		}
	}

	resize(frameSize, maxQueueSize, maxTrainingSamples);

	mFirstLoadConfig = false;
}

void NRRSNetwork::setStep(int step) {

	mLossStep = step;

	mNetworkLL2->updateLossHyperparams({{"step", step}});
	mLoss->update_hyperparams({{"step", step}});
	if (step == 3) {
		mPixelDebugCount = 0;
		cudaMemsetAsync(mPixelDebugBuffer, 0, sizeof(uint) * mResolutionSize,
						gpContext->cudaStream);
	}
}

void NRRSNetwork::updateLossOffset(uint offset) {
	// [TODO] may not true
	if (offset == 0) {
		++mPixelDebugCount;
	}
	mLoss->update_hyperparams({{"offset", offset}});
}

void NRRSNetwork::updateReliefErrorScale() {
	mLoss->update_hyperparams({{"update_error_scale", true}});
}

void NRRSNetwork::setDebugPixelForTraining(int debugPixelID) {
	mLoss->update_hyperparams({{"debug_pixel_id", debugPixelID}});
}

NAMESPACE_END(krr)
