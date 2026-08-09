#pragma once
#include <numeric>

#include "camera.h"
#include "file.h"
#include "renderpass.h"
#include "scene.h"
#include "window.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/optix.h"
#include "device/atomic.h"
#include "device/timer.h"

#include "train.h"
#include "nrrs.h"
#include "workqueue.h"
#include "myFilm.h"
#include "net/network.h"
#include "render/common/denoise/denoisetask.h"

NAMESPACE_BEGIN(krr)

class Film;

class NRRSPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<NRRSPathTracer>;
	KRR_REGISTER_PASS_DEC(NRRSPathTracer);

	NRRSPathTracer()  = default;
	~NRRSPathTracer() = default;

	void resize(const Vector2i &size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(RenderContext *context) override;
	void endFrame(RenderContext *context) override;
	void render(RenderContext *context) override;
	void renderUI() override;
	void finalize() override;

	// here, we overwrite the function from RenderPass::initialize()
	// so RenderApp::initialize() won't call this function
	// it's ok, because resize() will call this function
	void initialize(bool keepRenderedImage);

	string getName() const override { return "NRRSPathTracer"; }

	template <bool tIsTraining> void renderInternal(RenderContext *context);
	template <bool tIsTraining> void handleIntersections(const int depth);
	template <bool tIsTraining>
	KRR_DEVICE_FUNCTION void generateScatteredRays(const NRRSScatterRayWorkItem &w,
												   Sampler &sampler, const int depth,
												   const uint frameSizeWidth);
	template <bool tIsTraining> void handleEmissiveHit();
	template <bool tIsTraining> void handleMiss();
	void queryNetwork();
	void generateCameraRays(int sampleId);
	void traceClosest(const int depth);

	void traceShadow(const bool isTraining);

	template <bool tIsSimpleMode = false> void writeResultToRenderTarget(RenderContext *context);

	KRR_CALLABLE bool isInferMode(const int depth) {
		// TODO:[I] the original ADRRS: depth should > 0
		return mEnableRRS && RRS_CLAMP_MIN < RRS_CLAMP_MAX && depth <= mMaxRRSDepthIncluded &&
			   UBSNeedNetInfer(depth);
	}

	// RRS
	template <bool tIsSimpleMode, bool tIsTraining> void generateRRSNumber(const int depth);

	KRR_CALLABLE NRRSRayQueue *currentRayQueue(int depth) { return mRayQueue[depth & 1]; }
	KRR_CALLABLE NRRSRayQueue *nextRayQueue(int depth) { return mRayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

	// visiuallize
	KRR_DEVICE_FUNCTION RGB colorJetMap(float vIn, float vMax);
	uint loadArrayToInt(const char *arr, std::string &strConcated);

	// simple render
	void changeRenderMode();

	// guided path routines
	void resetTraining();
	void trainStep();
	void denoiseRenderedImage();

	void autoTrainControl();
	void saveAndExit();

	OptixBackend *mBackend;
	Camera::CameraData *mCamera{};
	LightSampler mLightSampler;

	// work queues
	NRRSRayQueue *mRayQueue[2]{}; // switching bewteen current and next queue
	NRRSMissRayQueue *mMissRayQueue{};
	NRRSHitLightRayQueue *mHitLightRayQueue{};
	NRRSShadowRayQueue *mShadowRayQueue{};
	NRRSScatterRayQueue *mScatterRayQueue{}; // bsdf evaluation (plus shadow ray generation)

	TidQueue *mScatterTidQueue{};
	NRRSInferenceQueue *mInferenceQueue{};

	uint *mScatterRayQueueSimplePixelIdPtr{nullptr};
	uint *mScatterRayQueuePixelIdPtr{nullptr};

	// weighted L
	RGB *mWeightedLBuffer{nullptr};
	RGB *mWeightedLBufferCurrent{nullptr};
	uint *mWeightedLBufferCurrentAcc{nullptr};
	float *mNumberSamplesThisPatch{nullptr};
	bool mFirstFrameUpdateWeightedL{true};
	float mWeightedLBlendWeight{0.5f};
	NRRSPixelStateBuffer *mPixelState;
	uint mMaxRateForPathNodesBuffer{4};
	NRRSPathNodesBuffer *mPathState;
	uint *mPathStateNodeIdxAtomicBuffer{nullptr}; // for remove duplicate nodes
	NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0> *mTrainBuffer;

	// global properties
	bool mDebugOutput{true};
	uint mDebugPixel{};
	bool mDebugTrainingData{false};
	int mMaxQueueSize;
	int mFrameId{0};

	// path tracing parameters
	int mSamplesPerPixel{1};
	int mSamplesPerPixelThisFrame{0};
	int mMaxDepth{6};
	float mFixedProbRR{1};
	bool mEnableNEE{true};
	bool mEnableClamp{false};
	float mClampMax{1e4f};
	int mRandomOffset{0}; // used to offset the random seed for each pixel

	uint mErrorImageScale = 2u;
	MyFilm *mRenderedImage{nullptr};
	RGB *mRenderedImageDenoised{nullptr};
	bool mRenderedImageDenoiseOnce{false};
	bool mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer{false};
	Film *mRefImageDebug{nullptr}; // TODO:[IM] now just for debug[for training RRS]

	// TODO, now only support mErrorImageScale = 1
	float *mErrorFactorXY{nullptr};
	float *mErrorFactorXX{nullptr};

	// cost for each pixel
	// float *mCostBuffer{nullptr};
	// uint mCostBufferAcc{0};

	bool mDrawRect{false};
	int mDrawRectSize{5};
	bool mRRSDivInRect{false};
	float mRRSDivInRectVal{1.0f};

	enum DenoiseInputType {
		Error,
		L
	} mDenoiseInputType{Error};
	std::shared_ptr<DenoiseTask> mDenoiseTask{nullptr};
	int mDenoiseFrames{-1};

	class ShowDebugBuffer {
	public:
		void renderUI(NRRSPathTracer *pass);

		inline bool showLi() const { return mShowType == ShowLi; }
		inline bool showRRS() const { return mShowType == ShowRRS; }
		inline bool showSpecular() const { return mShowType == ShowSpecular; }
		inline bool isNone() const { return mShowType == None; }

		void allocateBuffer(const uint resolutionSize);
		void freeBuffer();

		enum ShowType {
			ShowLi,
			ShowWeightedL,
			ShowRenderedImage,
			ShowRenderedImageDenoised,
			ShowRefImage, // [for debug]
			ShowRRS,
			ShowSpecular,
			ShowLError, // [for auto train]
			ShowWeightedLError,
			ShowErrorPerPixel,
			ShowTrainingSamplesPerPixel,
			ShowNumberSamplesPerPixel,
			ShowNetworkDebug,
			ShowErrorPerSample,
			None
		};

		enum ShowRRSMode {
			BeforeClamp,
			AfterClamp,
			Normalized,
			RealRay,
			Count
		};

		// show RRS
		ShowRRSMode mShowRRSMode{BeforeClamp};
		int mShowRRSWhichDepth{0}; // bit i is 1 means show depth i
		uint mShowRRSAcc{0};
		float *mShowRRSBuffer{nullptr};
		bool mResetRRSBufferEachFrame{false};

		// show specular rays
		uint mShowSpecularAcc{0};

		float *mShowBuffer{nullptr};

		// show Li
		int mShowLiType{0};
		bool mShowLiAddRadiance{true};
		bool mShowLiGray{false};
		int mShowLiDepth{0}; // query the network at this depth

		// show weighted L
		// show weighted L error
		// show error per pixel

		// show network debug
		bool mShowNetDebugTypeIsFloat{false};
		// int mShowNetDebugInt{0};// in network.cpp

		struct ShowErrorPerPixelParams {
			bool mLogScale{false};
			bool mLargeThan{false};
			float mLargeBound{1.0f};
			bool mShowRelativeToMean{false};
			float mShowRelativeToMeanScale{10.0f};
		} mShowErrorPerPixelParams;

		// general
		ShowType mShowType{ShowType::None};
		bool mCalcAvg;
		float *mSumGPUPtr{nullptr};
		float mSumCPU;

		// Jet
		bool mShowScalarJetOn{true};
		float mShowScalarJetMax{5.0f};
		float mShowScalarJetMaxUpBound{4.0f};
	} mShowBuffer;

	// Debug
	bool mDebugOn{false};
	int mDebugInt{0};
	bool mLowPowerMode{false};
	bool mLPMRender{false};
	bool mAdaptiveSamplingOn{false};
	int mAdaptiveSamplingRadius{50};
	int mAdaptiveSamplingSpp{2};

	// TODO: change to constexpr
	float RRS_CLAMP_MAX = 5.0f, RRS_CLAMP_MIN = 0.5f, RRS_NORMALIZE_RATE = 0.5f;
	bool mRRSNormalizeUseCeil{false};
	bool mEnableRRS{false};
	bool mDebugRRSNormalized{false}; // debug only, show whether normalized

	int mMaxRRSDepthIncluded{6};

	bool mCopyTheRRSNode{true};
	// if 2 training data are the same for RRSNet, ignore them[LL2Net still need them]
	bool mIgnoreSameTrainingDataForRRSNet{false};
	bool mDonnotNormalizeWhenLessThanOne{false};

	// RRS Scaler for each depth
	// constexpr static uint RS_MAX_DEPTH = 64;
	// bool mEnableRRSScaler{false};
	// bool mEnableRRSScalerSearch{false};
	// float *mRRSScaler{nullptr};

	// best rrs strategy
	constexpr static uint UBS_MAX_DEPTH = 64;
	bool mUseBestStrategy{false};
	bool mUBSSearch{false};
	float mUBSBestEfficiency{-1.0f};
	int *mUBSArray{nullptr}; // mUBSArray[i] means stratagy used in the depth i
	char *mUBSArrayChar{nullptr};
	int mUBSSearchState{-1}; // used when search
	int mUBSSearchStateMod{-1};
	int mUBSLastSearchState{-1};
	int mUBSMaxState{0};
	int mUBSBestState{0}; // when search is over, don't use this varaible
	bool mUBSTimerInitialized{false};
	float mUBSLastError{-1.0f};
	float mUBSLastDeltaTime{-1.0f};
	float *mUBSErrorBufferGPU{nullptr};

	CpuTimer::TimePoint mUBSLastTime{};

	enum ExpMethods {
		Exp_NRRS_MIX,
		Exp_NRRS,
		Exp_ADN,
		Exp_Count
	};

	// Experiments
	ExpMethods mExpMethods{};
	bool mExpOn{false};
	float *mExpRayCounter{nullptr}; // 10 seconds is ok
	float mExpTrainTime{0.0f};
	float mExpInferenceTime{0.0f};
	std::string mExpOutputFile{};
	int mExpState{0}; // 0: not initialized, 1: train, 2: inference
	CpuTimer::TimePoint mExpStartTime{};
	CpuTimer::TimePoint mExpCurrentTime{};
	Film *mExpImage{nullptr};
	bool mExpDenoiseAlways{false};

	void controlExp(RenderContext *context);

	bool UBSUseNetOutputRRS(const int depth) const {
		// [TODO] delete it when release
		bool notDesired = false;
		notDesired |= !(mUseBestStrategy || mUBSSearch);

		int state = 0;
		if (mUBSSearch) {
			int level = 1;
			for (int i = 0; i < depth; ++i) {
				level *= RRSStarategy::Count;
			}
			state = (mUBSSearchState / level) % RRSStarategy::Count;
		} else if (mUseBestStrategy) {
			state = mUBSArray[depth];
		}

		notDesired |= (state == RRSStarategy::FixedRR);
		if (notDesired) {
			Log(Warning, "[%s::%d] This is not desired, state: %d", __FILE__, __LINE__, state);
		}

		return state == RRSStarategy::NRRS;
	}

	bool UBSNeedNetInfer(const int depth) const {
		if (mUseBestStrategy) {
			return mUBSArray[depth] != RRSStarategy::FixedRR;
		}
		if (mUBSSearch) {
			int level = 1;
			for (int i = 0; i < depth; ++i) {
				level *= RRSStarategy::Count;
			}
			int state = (mUBSSearchState / level) % RRSStarategy::Count;
			return state != RRSStarategy::FixedRR;
		}
		// default don not change anything
		return true;
	}

	void UBSUpdateAccordingArrayChar() {
		for (int i = 0; i < mMaxDepth; ++i) {
			int num = mUBSArrayChar[i];
			num -= int('0');
			if (num >= 0 && num < RRSStarategy::Count) {
				mUBSArray[i] = RRSStarategy(num);
			} else {
				mUBSArrayChar[i] = '0';
			}
		}
		mUBSArrayChar[mMaxDepth] = '\0';
	}

	void UBSResetAndBeginSearch() {
		mUBSBestEfficiency	= 0.0f; // [TODO] whether should reset?
		mUBSLastError		= -1.0f;
		mUBSLastDeltaTime	= -1.0f;
		mUBSSearch			= true;
		mUseBestStrategy	= false;
		mUBSSearchState		= -1;
		mUBSSearchStateMod	= -1;
		mUBSLastSearchState = -1;
		mUBSMaxState		= (int) pow((float) RRSStarategy::Count, (float) mMaxDepth);

		// UBS Heuristic
	}

	void UBSEndSearch() {
		mUBSTimerInitialized = false;
		mUBSSearch			 = false;
		// update best
		std::string bsStr = "Best Strategy: ";
		int strategy	  = mUBSBestState;
		for (int i = 0; i < mMaxDepth; ++i) {
			int now			 = strategy % RRSStarategy::Count;
			mUBSArray[i]	 = RRSStarategy(now);
			mUBSArrayChar[i] = '0' + now;
			strategy		 = strategy / RRSStarategy::Count;
			bsStr += std::to_string(now) + ((i == mMaxDepth - 1) ? "" : ", ");
		}
		mUBSArrayChar[mMaxDepth] = '\0';
		Log(Info, "%s Efficiency = %f", bsStr.c_str(), mUBSBestEfficiency);
	}

	std::string UBSState2String(int state) {
		std::string s = "";
		int strategy  = state;
		for (int i = 0; i < mMaxDepth; ++i) {
			int now	 = strategy % RRSStarategy::Count;
			strategy = strategy / RRSStarategy::Count;
			s += std::to_string(now) + ((i == mMaxDepth - 1) ? "" : ", ");
		}
		return s;
	}

	std::string UBSGetBestStateString() {
		std::string s = "";
		for (int i = 0; i < mMaxDepth; ++i) {
			int now = mUBSArray[i];
			s += std::to_string(now) + ((i == mMaxDepth - 1) ? "" : ", ");
		}
		return s;
	}

	void UBSRecordEfficiency(float efficiency);

	bool UBSNextState();

	void UBSAnalyseFrame();

	enum RRSStarategy {
		FixedRR = 0,
		ADN		= 1,
		NRRS	= 2,
		Count	= 3
	};

	// ADN parameters
	class NRRSParams {
	public:
		KRR_HOST void renderUI(NRRSPathTracer *pass);

		uint mBatchPerFrame{5};
		static const uint mBatchSize{NRRS_TRAIN_BATCH_SIZE};

		// simple render, payload is small
		bool mSimpleRenderOn{false};

		bool mStopTraining{false};
		bool mTrainOneStep{false};
		bool mTrainOneBatch{false};

		float *mRRSArray{nullptr};
		float *mRRSArrayCeil{nullptr}; // for calculate the sum
		float *mTempGPUBuffer{nullptr};
		constexpr static uint mTempGPUBufferSize = 2048; // [note] only the 32*4 bytes can be used
		float *mTempCPUBuffer{nullptr};
		constexpr static uint mTempCPUBufferSize = mTempGPUBufferSize;

		bool mUseWeightWindow = false;

		constexpr static uint mSumPosInGPUBuffer		  = 0;
		constexpr static uint mSpecularTidSizeInGPUBuffer = 1; // uint = float
		constexpr static uint mUBSErrorSumPosInGPUBuffer  = 2; // float
															   // next is 3

		constexpr static uint mInferQueueSizeInCPUBuffer  = 0;
		constexpr static uint mSpecularTidSizeInCPUBuffer = 1; // uint = float
		constexpr static uint mPathStateSizeInCPUBuffer	  = 2; // uint = float
															   // next is 3

		std::shared_ptr<NRRSNetwork> mNet;

		// is UBS on, this arg have no effects
		bool mLossUseNetworkOutputRRS{false};

		// training data
		bool mIgnoreZeroL{false};

	} mNRRSParams;

	// auto train
	bool mAutoTrain{false};
	float mAutoTrainTime1FramesInSeconds{1.0f};
	int mAutoTrainRequestExit{0};
	float mAutoTrainStateTime[4] = {10, 10, 10, 60};

	friend void to_json(json &j, const NRRSPathTracer &p) {
		j = json{
			{"nee", p.mEnableNEE},
			{"max_depth", p.mMaxDepth},
			{"rr", p.mFixedProbRR},
			{"enable_clamp", p.mEnableClamp},
			{"clamp_max", p.mClampMax},
			{"batch_per_frame", p.mNRRSParams.mBatchPerFrame},
			{"batch_size", p.mNRRSParams.mBatchSize},
			{"rrs_clamp_min", p.RRS_CLAMP_MIN},
			{"rrs_clamp_max", p.RRS_CLAMP_MAX},
			{"rrs_normalize_rate", p.RRS_NORMALIZE_RATE},
			{"spp", p.mSamplesPerPixel},
			{"max_rate_for_path_nodes_buffer", p.mMaxRateForPathNodesBuffer},
			{"blend_weight_for_error", p.mWeightedLBlendWeight},
			{"max_depth_to_use_rrs_included", p.mMaxRRSDepthIncluded},
			{"copy_the_rrs_node", p.mCopyTheRRSNode},
			{"random_offset", p.mRandomOffset},
		};
	}

	friend void from_json(const json &j, NRRSPathTracer &p) {
		p.mRandomOffset				 = j.value("random_offset", 0);
		p.mEnableNEE				 = j.value("nee", true);
		p.mMaxDepth					 = j.value("max_depth", 6);
		p.mFixedProbRR				 = j.value("rr", 0.8);
		p.mEnableClamp				 = j.value("enable_clamp", false);
		p.mClampMax					 = j.value("clamp_max", 1e4f);
		p.mNRRSParams.mBatchPerFrame = j.value("batch_per_frame", 5);
		// p.mNRRSParams.mBatchSize	 = j.value("batch_size", NRRS_TRAIN_BATCH_SIZE);
		p.RRS_CLAMP_MIN				 = j.value("rrs_clamp_min", 0.5f);
		p.RRS_CLAMP_MAX				 = j.value("rrs_clamp_max", 5.0f);
		p.RRS_NORMALIZE_RATE		 = j.value("rrs_normalize_rate", 0.5f);
		p.mRRSNormalizeUseCeil		 = j.value("rrs_normalize_use_ceil", false);
		p.mSamplesPerPixel			 = j.value("spp", 1);
		p.mMaxRateForPathNodesBuffer = j.value("max_rate_for_path_nodes_buffer", 4);
		p.mWeightedLBlendWeight		 = j.value("blend_weight_for_error", 0.5f);
		p.mMaxRRSDepthIncluded		 = j.value("max_depth_to_use_rrs_included", 10);
		p.mCopyTheRRSNode			 = j.value("copy_the_rrs_node", false);
		p.mIgnoreSameTrainingDataForRRSNet =
			j.value("ignore_same_training_data_for_rrs_net", false);
		p.mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer =
			j.value("use_acc_buffer_not_denoised_nrrs", false);

		// debug
		p.mShowBuffer.mShowNetDebugTypeIsFloat = j.value("show_net_debug_type_is_float", false);

		p.mNRRSParams.mStopTraining		  = j.value("stop_training", false);
		p.mDonnotNormalizeWhenLessThanOne = j.value("donnot_normalize_when_less_than_one", true);

		p.mAutoTrain = j.value("nrrs_auto_train", false);
		if (p.mAutoTrain) {
			p.mAutoTrainStateTime[0] = j.value("nrrs_auto_train_state0_time", 10);
			p.mAutoTrainStateTime[1] = j.value("nrrs_auto_train_state1_time", 10);
			p.mAutoTrainStateTime[2] = j.value("nrrs_auto_train_state2_time", 10);
			p.mAutoTrainStateTime[3] = j.value("nrrs_auto_train_state3_time", 60);
		}

		p.mErrorImageScale = j.value("error_image_scale", 2u);

		std::string weightsPath = j.value("weights_path", "");

		// Experiment: from global config
		const auto globalConfig = gpContext->globalConfig;
		p.mExpOn				= globalConfig.value("exp_on", false);
		p.mExpTrainTime			= globalConfig.value("exp_train_time", 60.0f);
		p.mExpInferenceTime		= globalConfig.value("exp_inference_time", 60.0f);
		p.mExpOutputFile		= globalConfig.value("exp_output_file", "result.exr");
		p.mExpState				= 0;
		p.mExpDenoiseAlways		= globalConfig.value("exp_denoise_always", false);

		bool updateLR = false;
		if (p.mExpOn) {
			std::string methods = j.value("rrs_method", "");
			if (methods == "nrrs+") {
				p.mExpMethods = ExpMethods::Exp_NRRS_MIX;
			} else if (methods == "nrrs") {
				p.mExpMethods = ExpMethods::Exp_NRRS;
			} else if (methods == "adn") {
				p.mExpMethods = ExpMethods::Exp_ADN;
				updateLR	  = true;
			} else {
				Log(Fatal, "[Exp] Methods Error: %s", methods);
			}
		}

		bool net1AidNet2	  = j.value("net1_aid_net2", false);
		bool ll2UseTwoHead	  = j.value("ll2_use_two_head", false);
		int ll2UseTwoHeadType = j.value("ll2_two_head_type", 1);

		if (j.contains("nrrs_config")) {
			string config_path = j.at("nrrs_config");
			std::ifstream f(config_path);
			if (f.fail()) {
				Log(Fatal, "Open network config file failed!");
			}
			json nn_config = json::parse(f, nullptr, true, true)["nn"];

			if (updateLR) { // make adn more robust to anti nan
				auto &tmp_config			= nn_config["level0"]["optimizer"]["nested"];
				tmp_config["learning_rate"] = tmp_config.value<float>("learning_rate", 0) / 5.0f;
			}

			// pass to network
			nn_config["error_image_scale"] = p.mErrorImageScale;
			nn_config["net1_aid_net2"]	   = net1AidNet2;
			nn_config["ll2_use_two_head"]  = ll2UseTwoHead;
			nn_config["ll2_two_head_type"] = ll2UseTwoHeadType;

			p.mNRRSParams.mNet = std::make_shared<NRRSNetwork>(nn_config);
			p.mNRRSParams.mNet->setWeightsLoadPath(weightsPath);
		} else {
			Log(Fatal, "Network config do not specified!");
		}

		// denoise
		const bool noAlbedo	  = j.value("nrrs_denoise_dont_need_albedo", false);
		const bool denoiseOn  = j.value("denoise_on", false);
		p.mDenoiseFrames	  = j.value("denoise_frames", -1);
		p.mDenoiseTask		  = std::make_shared<DenoiseTask>();
		auto denoiseTaskParam = j.value("denoise_params", json{});
		denoiseTaskParam.update({{"enable", denoiseOn}}, true);
		p.mDenoiseTask->from_json(denoiseTaskParam);
		p.mDenoiseTask->setDontNeedAlbedo(noAlbedo);
	}

	// simple render part
public:
	void renderSimple(RenderContext *context);
	void generateCameraRaysSimple(int sampleId);
	KRR_CALLABLE SWPTRayQueue *currentRayQueueSimple(int depth) {
		return mRayQueueSimple[depth & 1];
	}
	KRR_CALLABLE SWPTRayQueue *nextRayQueueSimple(int depth) {
		return mRayQueueSimple[(depth & 1) ^ 1];
	}

	void traceClosestSimple(const int depth);
	void traceShadowSimple();
	void handleEmissiveHitSimple();
	void handleMissSimple();
	void handleIntersectionsSimple(const int depth);
	KRR_DEVICE_FUNCTION void generateScatteredRaysSimple(const SWPTScatterRayWorkItem &w,
														 Sampler &sampler, const int depth);

	// work queues for simple render
	SWPTRayQueue *mRayQueueSimple[2]{};
	SWPTMissRayQueue *mMissRayQueueSimple{};
	SWPTHitLightRayQueue *mHitLightRayQueueSimple{};
	SWPTShadowRayQueue *mShadowRayQueueSimple{};
	SWPTScatterRayQueue *mScatterRayQueueSimple{};
};

NAMESPACE_END(krr)