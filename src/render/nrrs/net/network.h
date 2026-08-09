#pragma once

#include "networkcommon.h"
#include "myNetworkLL2.h"
#include "../myFilm.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

enum ShowLossType {
	None,
	All,
	LL2_Loss,
	RRS_Loss,
	RRS_GRAD_ALL, // 4
	RRS_GRAD_AVG,
	RRS_GRAD_MIN,
	RRS_GRAD_RRS,
	Count,
};

class NRRSNetwork {
public:
	NRRSNetwork(json &config);
	~NRRSNetwork();

	void renderUI();

	void loadWeights(const std::string &filePath);
	void saveWeights(const std::string &filePath);

	float train(GPUMatrix_Float &input, GPUMatrix_Float &output, const int offsetOfTrainingData,
				const bool onlyLL2);
	// note that prepareInferenceData() should match inference(); onlyRRSNet = != onlyLL2Net
	// [#1] Query
	//   [1.1] query L/L2 : onlyRRSNet = false, onlyLL2Net = true
	//   [1.2] gen RRS    : onlyRRSNet = true , onlyLL2Net = false
	// that is to say, when AID=true, we can not obtain both L/L2 and RRS in one inference (useless)
	void prepareInferenceData(const int inferenceQueueSize, const NRRSInferenceQueue *inferQueue,
							  const AABB &sceneAABB, bool onlyRRSNet, MyFilm *renderedImage,
							  const uint *pixelID);
	void inference(const int inferenceQueueSize, const bool onlyLL2Net,
					  const NRRSInferenceQueue *inferQueue, MyFilm *renderedImage,
					  const uint *pixelID);

	void resize(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples);
	void reset(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples);

	void getLossHelper(float *&thp, float *&errorForTraining, float *&errorForAvgForTraining,
					   float *&refMean, float *&sampleWeight, float *&errorPerPixel,
					   float *&numSamplesForTraining, uint *&pixelID) const {
		thp					   = mThpForTraining;
		errorForTraining	   = mErrorForTraining;
		errorForAvgForTraining = mErrorForAvgForTraining;
		refMean				   = mRefMeanForTraining;
		sampleWeight		   = mSampleWeightForTraining;
		pixelID				   = mPixelIDForTraining;
		numSamplesForTraining  = mNumberSamplesForTraining;

		errorPerPixel = mErrorPerPixel;
	}

	float *getErrorPerPixel() { return mErrorPerPixel; }

	void getPixelDebugBuffer(uint *&ptr, uint &cnt) {
		ptr = mPixelDebugBuffer;
		cnt = mPixelDebugCount;
	}

	int getStep() const { return mLossStep; }

	void setStep(int step);

	bool getClampPixelError() const { return mClampPixelError; }
	bool getPixelErrorMultiplySamples() const { return mPixelErrorMultiplySamples; }

	void updateLossOffset(uint offset);
	void updateReliefErrorScale();

	void setShowLossIndexLL2() { mLossShowLossIndex = ShowLossType::LL2_Loss; }

	void setDebugPixelForTraining(int debugPixelID);

	float *getLossSumErrorGPUPtr() { return mLossSumErrorGPUPtr; }

	// precision_t *getInferenceOutputBufferPtr0()
	// const { 	return mNetworkLL2->getInferenceOutputBufferPtr_L();
	// }

	NRRSNetworkOutputTraining1 *getTrainingRRSNetOutputBufferPtr() {
		return mTrainingOutputBuffer1.data();
	}

	// inference buffer
	float *getInferenceInputBufferPtr0() const { return mNetworkLL2->getInferenceInputBufferPtr(); }
	precision_t *getInferenceInputBufferPtr1() const { return mInferenceInputBuffer1.data(); }

	precision_t *getInferenceOutputBufferPtr_L() const {
		return mNetworkLL2->getInferenceOutputBufferPtr_L();
	}
	precision_t *getInferenceOutputBufferPtr_L2() const {
		return mNetworkLL2->getInferenceOutputBufferPtr_L2();
	}

	precision_t *getInferenceOutputBufferPtr1() const { return mInferenceOutputBuffer1.data(); }
	void getInferenceInputOutputBufferPtr0(float *&input, precision_t *&output_L,
										   precision_t *&output_L2) const {
		input	  = getInferenceInputBufferPtr0();
		output_L  = getInferenceOutputBufferPtr_L();
		output_L2 = getInferenceOutputBufferPtr_L2();
	}

	void getInferenceInputOutputBufferPtr1(precision_t *&input, precision_t *&output) {
		input  = mInferenceInputBuffer1.data();
		output = mInferenceOutputBuffer1.data();
	}

	bool getTrainSigmaOrX2() const { return mTrainSigmaOrX2; }
	void setWeightsLoadPath(const std::string &path) { mWeightsLoadPath = path; }
	void setScene(std::shared_ptr<Scene> scene) { mScene = scene; }

	float gamma1() const { return mGamma1; }
	float gamma2() const { return mGamma2; }
	float gamma3() const { return mGamma3; }
	float gamma4() const { return mGamma4; }

private:
	CudaGraph mCuGraphRRSNet;

	uint mResolutionSize{0};
	uint mMaxQueueSize{0};
	uint mMaxTrainingSamples{0};

	uint mErrorImageScale{1u};

	std::shared_ptr<Scene> mScene{nullptr};

	// training buffers / memories

	// owned by LL2Net
	// GPUMemory<float> mInferenceInputBuffer0;
	// GPUMemory<precision_t> mInferenceOutputBuffer0;

	// maybe NRRSNetworkInput1 or NRRSNetworkInput1AID
	GPUMemory<precision_t> mInferenceInputBuffer1;
	GPUMemory<precision_t> mInferenceOutputBuffer1;

	// maybe NRRSNetworkInput1 or NRRSNetworkInput1AID
	GPUMemory<float> mTrainingInputBuffer1;
	GPUMemory<NRRSNetworkOutputTraining1> mTrainingOutputBuffer1;

	// loss
	bool mLossClampOn{false};
	float mLossClampMax{500.0f};
	int mLossStep{2};
	ShowLossType mLossShowLossIndex{All};
	float *mLossSumErrorGPUPtr{nullptr};
	bool mTrainSigmaOrX2{true}; // train E[x^2] if false

	// a debug int pass to loss
	int mShowNetworkDebugInt{0};

	float mGamma1{1e-1f};
	float mGamma2{1e-1f};
	float mGamma3{1e-2f};
	float mGamma4{0.5f};

	// training: size = maxTrainingSamples * typeSize
	float *mThpForTraining{nullptr};
	float *mErrorForTraining{nullptr};
	float *mErrorForAvgForTraining{nullptr};
	float *mRefMeanForTraining{nullptr}; // TODO: this may not needed
	float *mSampleWeightForTraining{nullptr};
	uint32_t *mPixelIDForTraining{nullptr};
	float *mNumberSamplesForTraining{nullptr};

	// batch size * 6
	precision_t *mLL2Ptr{nullptr};

	// pixels size
	float *mErrorPerPixel{nullptr};		  // 2x pixel size, record error1 and error1/sqrt(?)
	uint32_t *mPixelDebugBuffer{nullptr}; // for debug
	uint32_t mPixelDebugCount{0};
	bool mResetPixelDebugBufferEachFrame{false};
	float mDebugFloat{0.01f};

	bool mPixelErrorMultiplySamples{false};
	bool mClampPixelError{true};
	bool mDebugLoss{false};

	// [RRSNet] when after long training and error is not going down, we reduce the error scale
	bool mReliefError{false};

	std::string mWeightsLoadPath;

	json mConfig;
	bool mFirstLoadConfig{true};

	bool mNetIsInitialized{false};
	std::shared_ptr<MyNetworkLL2> mNetworkLL2{nullptr};
	std::shared_ptr<Network_RRS> mNetworkRRS{nullptr};
	std::shared_ptr<Optimizer> mOptimizer{nullptr};
	std::shared_ptr<Loss> mLoss{nullptr};
	std::shared_ptr<Trainer_RRS> mTrainerRRS{nullptr};

	// AID
	bool mNet1AID{false}; // net0 only for training if true
	std::shared_ptr<NetworkWithInputEncoding> mNetworkAID{nullptr};
	std::shared_ptr<Trainer_RRS_AID> mTrainerRRSAID{nullptr};

public:
	static const int LL2_NET{0};
	static const int RRS_NET{1};
};

NAMESPACE_END(krr)