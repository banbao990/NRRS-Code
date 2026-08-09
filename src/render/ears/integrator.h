#pragma once

#include "common.h"
#include "render/common/tree/tree.h"
#include "workqueue.h"
#include "util/task.h"
#include "render/common/commonworkqueue.h"
#include "render/common/tree/treenodetypes.h"
#include "octree.h"
#include "render/passes/denoise/denoise.h"
#include "render/common/commoncudautilshost.h"
#include "render/common/denoise/denoisetask.h"

#include "NNcommon.h"
#include "render/nrc/nrctrain.h"

NAMESPACE_BEGIN(krr)

class Film;
typedef RadianceTreeNode ADRRSTreeNodeType;

// NN Cache
// position[3], incoming direction[2], roughness[1]
struct EARSNetworkInput {
	Vector3f mPos;	  // normalized pos to [0, 1]^3
	Vector2f mDir;	  // normalized dir with 1-norm
	float mRoughness; // roughness
};

// L[3], L^2[3], Cost
struct EARSNetworkOutput {
	RGB mL;
	RGB mVar;
	float mCost;
};

constexpr uint32_t LL2NET_DIM_INPUT			= sizeof(EARSNetworkInput) / sizeof(float);
constexpr uint32_t LL2NET_DIM_OUTPUT		= sizeof(EARSNetworkOutput) / sizeof(float);
constexpr uint32_t LL2NET_DIM_OUTPUT_PADDED = 16;

class EARSPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<EARSPathTracer>;
	KRR_REGISTER_PASS_DEC(EARSPathTracer);

	enum RRSKind {
		ADRRS,
		EARS,
		Count,
	};

	EARSPathTracer()  = default;
	~EARSPathTracer() = default;
	void initialize();

	template <typename... Args> KRR_DEVICE void debugPrint(const char *fmt, Args &&...args);
	template <typename... Args> KRR_DEVICE void debugPrintOneCall(const char *fmt, Args &&...args);

	void resize(const Vector2i &size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(RenderContext *context) override;
	void endFrame(RenderContext *context) override;
	void render(RenderContext *context) override;
	void renderUI() override;
	void finalize() override; /* Save the rendering (of the last iter) maybe more. */

	void renderInternal(RenderContext *context); // for visiualization

	string getName() const override { return "EARS Path Tracer"; }

	void handleHit();
	void handleMiss();
	template <bool tIsTraining> void handleIntersections(const int depth);
	template <bool tIsTraining>
	KRR_DEVICE_FUNCTION void generateScatteredRays(EARSScatterRayWorkItem &w, Sampler &sampler,
												   const int depth);

	KRR_CALLABLE EARSRayQueue *mCurrentRayQueue(int depth) { return mRayQueue[depth & 1]; }
	KRR_CALLABLE EARSRayQueue *mNextRayQueue(int depth) { return mRayQueue[(depth & 1) ^ 1]; }

	void updateOctree();

	void queryRadiance();
	void traceClosest(int depth);
	void generateCameraRays(const int sampleId);
	void traceShadow();

	void generateRRSNumber(const int depth);

	// EARS
	void postProcessAfterInteration(RenderContext *context, bool denoiseForUITest);
	void updateImageStatistics();

	uint loadArrayToInt(const char *arr, std::string &strConcated);
	KRR_DEVICE_FUNCTION RGB colorJetMap(float vIn, float vMax);

	void debugInfosOutput();

	int mRandomOffset{0}; // used to offset the random seed for each pixel

	/* The target distribution (radiance or radiance * bsdf). */
	int mIter{0}; /* How many iteration have passed? Each iteration, the pass should be doubled.*/
	int mSppPerIteration{64}; /* A "pass" is some number of frames. */

	int mTwiceIterInterval{8u};

	/* The following state parameters are used in offline setup with a given budget. */
	void resetOctree();			   /* Reset the SD-Tree to the beginning. */
	int mTrainingIterations{-1};   /* The number of iterations for training (-1 means depends on
									  budget) */
	bool mAutoBuild{false};		   /* Automatically rebuild if the current render pass finishes. */
	Film *mRenderedImage{nullptr}; /* The image currently being rendered. @addition VAPG */
	uint mRenderedImageSpp{0u};
	bool mClearRenderedImageEachFrame{false};

	// Basic
	OptixBackend *mBackend{};
	Camera::CameraData *mCamera{};
	LightSampler mLightSampler;

	// PT
	int mMaxQueueSize;
	int mSpp{1};
	int mMaxDepth{10};
	float mProbRR{1.0};
	bool mEnableNEE{true};
	bool mEnableClamp{false};
	float mClampMax{1e3f};

	// Debug
	bool mDebugOn{false};
	int mDebugIntNumber = 1564;
	StatMinMaxAvgGPU *mDebugStats{nullptr};

	enum ShowRRSMode {
		BeforeClamp,
		AfterClamp,
		Normalized,
		RealRay,
		RRSModeCount
	};

	bool mShowRRS{false};
	ShowRRSMode mShowRRSMode{BeforeClamp};
	int mShowRRSAcc{0u};
	bool mResetRRSBufferEachFrame;
	float *mShowRRSBuffer{nullptr};
	int mShowRRSWhichDepth{0}; // bit i is 1 means show depth i
	bool mShowScalarJetOn{true};
	float mShowScalarJetMax{5.0f};
	float mShowScalarJetMaxUpBound{8.0f};

	// GBuffer
	bool mShowGBuffer{false};
	uint mGBufferKind{0};

	// queues
	EARSRayQueue *mRayQueue[2]{nullptr, nullptr}; // switching bewteen current and next queue
	EARSMissRayQueue *mMissRayQueue{nullptr};
	EARSHitLightRayQueue *mHitLightRayQueue{nullptr};
	EARSShadowRayQueue *mShadowRayQueue{nullptr};
	EARSScatterRayQueue *mScatterRayQueue{nullptr};
	EARSPixelStateBuffer *mPixelState;

	bool mEnableLearning{false};
	bool mEnableRRS{false};
	bool mEnableRRSAfterPreTraining{false};
	int mPreTrainingIterations = 3; // ears default setting
	RRSKind mRRSMethod{RRSKind::EARS};
	uint mMaxRateForPathNodesBuffer{4};
	EARSPathNodesBuffer *mPathState{};

	// ADRRS/EARS
	// TODO: change to constexpr
	static constexpr uint INFER_MAX_QUEUE_SIZE_RATE = 2;
	float RRS_CLAMP_MAX = 5.0f, RRS_CLAMP_MIN = 0.5f, RRS_NORMALIZE_RATE = 0.5f;
	bool mRRSNormalizeUseCeil{false};
	bool mShowLi{false};
	bool mShowLiAddRadiance{true};
	int mShowLiType{0}; // 0: Li, 1: LiMean, 2: Li^2, 3: cost[NN]
	float mShowLiCostWeightCoeff{0.2f};
	bool mShowLiSamplingNode{false};
	int mShowLiLookUpType{0}; // 0: noraml, 1: ignore invalid node, 2: min depth
	int mShowLiLookUpMaxDepth{0};
	int mShowLiDepth{0};
	float *mRRSArray{nullptr};
	float *mRRSArrayCeil{nullptr};

	bool mKeepLastIterStatistics{false};

	bool mDonnotNormalizeWhenLessThanOne{false};

	bool mDiff2RemoveTopK		   = true;
	int mTopK					   = 1000;	// 0.1%
	bool mUseCurrentImageStatistic = false; // defualt false, ears use last iter

	float *mTempGPUBuffer{nullptr};
	constexpr static uint mTempGPUBufferSize	  = 2048; // [note] only the 32*4 floats can be used
	constexpr static uint mRRSSumPosInGPUBuffer	  = 0;
	constexpr static uint mDiff2SumPosInGPUBuffer = 1;
	constexpr static uint mCostSumPosInGPUBuffer  = 2;
	constexpr static uint mTopKSumPosInGPUBuffer  = 3;
	constexpr static uint mUBSErrorSumPosInGPUBuffer = 4;

	constexpr static uint mTempPosInGPUBuffer = 32 * 4; // warpsize * 4
	float *mTempCPUBuffer{nullptr};
	constexpr static uint mTempCPUBufferSize			   = mTempGPUBufferSize;
	constexpr static uint mScatterRayQueueSizeInCPUBuffer  = 0;
	constexpr static uint mNonSpecularQueueSizeInCPUBuffer = 1;
	constexpr static uint mEARSCostGPUPointerInCPUBuffer   = 2;
	constexpr static uint mPosInCPUBufferOff1 = sizeof(FloatPointerWarpper) / sizeof(float);

	Octree *mOctree{nullptr};
	static constexpr uint mOctreeInitDepth = 2; // ears default 2
	int mOctreeMaxMemory{24};					// Max memory in megabytes.

	TidQueue *mScatterTidQueue{nullptr};
	EARSInferenceQueue *mNonSpecularTidQueue{nullptr};
	bool mShowRenderedImage{false};

	std::shared_ptr<DenoiseTask> mDenoiseTask{nullptr};
	bool mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer{false};
	struct EARSImageStatistic *mImageStatistic{nullptr};

	// [1] regenerateImageStatistics()
	float *mTempResolutionSizeBuffer{nullptr};

	// Experiments
	bool mExpOn{false};
	bool mExpUseUBS{false};
	float *mExpRayCounter{nullptr}; // 10 seconds is ok
	float mExpTrainTime{0.0f};
	float mExpInferenceTime{0.0f};
	std::string mExpOutputFile{};
	int mExpState{0}; // 0: not initialized, 1: train, 2: inference
	CpuTimer::TimePoint mExpStartTime{};
	CpuTimer::TimePoint mExpCurrentTime{};
	Film *mExpImage{nullptr};

	// NN cache
	bool mUseNNCache{false};
	bool mUseNNCacheWithTree{false}; // mUseNNCache = false => mUseNNCacheWithTree = false
	bool mShowLi_NNCache{true};

	json mConfig{};
	bool mFirstLoadConfig{true};
	bool mNetIsInitialized{false};
	int mMaxTrainingSamples{0};
	int mShowLossIndex{1}; // 0: none; 1: all; 2: L; 3: L^2; 4: cost

	uint mBatchPerFrame{5};
	static const uint mBatchSize{65'536 * 8};

	bool mLossClampOn{false};
	float mLossClampMax{500.0f};
	bool mTrainSigmaOrX2{true}; // train E[x^2] if false
	int mLossShowLossIndex{1};	// 0: none; 1: all

	std::shared_ptr<NetworkWithInputEncoding> mNetwork{nullptr};
	std::shared_ptr<Optimizer> mOptimizer{nullptr};
	std::shared_ptr<Loss> mLoss{nullptr};
	std::shared_ptr<Trainer_LL2> mTrainer{nullptr};

	NetworkTrainBuffer<EARSNetworkInput, EARSNetworkOutput> *mTrainBuffer{nullptr};

	// best rrs strategy
	enum E_RRSStarategy {
		E_FixedRR = 0,
		E_ADRRS	  = 1,
		E_EARS	  = 2,
		E_Count	  = 3
	};

	bool shouldGenRRS(int depth) { return mEnableRRS && UBSNeedNetInfer(depth); }

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

	bool UBSUseNetOutputRRS(const int depth) const {
		// [TODO] delete it when release
		bool notDesired = false;
		notDesired |= !(mUseBestStrategy || mUBSSearch);

		int state = 0;
		if (mUBSSearch) {
			int level = 1;
			for (int i = 0; i < depth; ++i) {
				level *= E_RRSStarategy::E_Count;
			}
			state = (mUBSSearchState / level) % E_RRSStarategy::E_Count;
		} else if (mUseBestStrategy) {
			state = mUBSArray[depth];
		}

		notDesired |= (state == E_RRSStarategy::E_FixedRR);
		if (notDesired) {
			Log(Warning, "[%s::%d] This is not desired, state: %d", __FILE__, __LINE__, state);
		}

		return state == E_RRSStarategy::E_EARS;
	}

	bool UBSNeedNetInfer(const int depth) const {
		if (mUseBestStrategy) {
			return mUBSArray[depth] != E_RRSStarategy::E_FixedRR;
		}
		if (mUBSSearch) {
			int level = 1;
			for (int i = 0; i < depth; ++i) {
				level *= E_RRSStarategy::E_Count;
			}
			int state = (mUBSSearchState / level) % E_RRSStarategy::E_Count;
			return state != E_RRSStarategy::E_FixedRR;
		}
		// default don not change anything
		return true;
	}

	void UBSUpdateAccordingArrayChar() {
		for (int i = 0; i < mMaxDepth; ++i) {
			int num = mUBSArrayChar[i];
			num -= int('0');
			if (num >= 0 && num < E_RRSStarategy::E_Count) {
				mUBSArray[i] = E_RRSStarategy(num);
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
		mUBSMaxState		= (int) pow((float) E_RRSStarategy::E_Count, (float) mMaxDepth);

		// UBS Heuristic
	}

	void UBSEndSearch() {
		mUBSTimerInitialized = false;
		mUBSSearch			 = false;
		// update best
		std::string bsStr = "Best Strategy: ";
		int strategy	  = mUBSBestState;
		for (int i = 0; i < mMaxDepth; ++i) {
			int now			 = strategy % E_RRSStarategy::E_Count;
			mUBSArray[i]	 = E_RRSStarategy(now);
			mUBSArrayChar[i] = '0' + now;
			strategy		 = strategy / E_RRSStarategy::E_Count;
			bsStr += std::to_string(now) + ((i == mMaxDepth - 1) ? "" : ", ");
		}
		mUBSArrayChar[mMaxDepth] = '\0';
		Log(Info, "%s Efficiency = %f", bsStr.c_str(), mUBSBestEfficiency);
	}

	std::string UBSState2String(int state) {
		std::string s = "";
		int strategy  = state;
		for (int i = 0; i < mMaxDepth; ++i) {
			int now	 = strategy % E_RRSStarategy::E_Count;
			strategy = strategy / E_RRSStarategy::E_Count;
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

	void UBSRecordEfficiency(float efficiency) {
		// Log(Info, "state: %s, efficiency: %f", UBSState2String(mUBSLastSearchState).c_str(),
		// efficiency);

		if (efficiency > mUBSBestEfficiency) {
			mUBSBestEfficiency = efficiency;
			// attention, this is last search state
			mUBSBestState = mUBSLastSearchState;
		}
	}

	bool UBSNextState() {
		static int sUBSCnt = 0;
		// now just enumerate all state
		// do one more for last turn
		// mUBSMaxState = 0..01, so run one more when 1...10 to record 1...10
		if (mUBSSearchState < mUBSMaxState) {
			mUBSLastSearchState = mUBSSearchState;
			if (mUBSSearchState == 0) {
				sUBSCnt = 0;
			}
			++mUBSSearchStateMod;
			if (mUBSSearchStateMod == 729) {
				++sUBSCnt;
				mUBSSearchState = mUBSBestState;
				Log(Info, "Best State in Stage %d: %s", sUBSCnt,
					UBSState2String(mUBSBestState).c_str());
			}
			mUBSSearchState += int(pow(729, sUBSCnt));
			return true;
		}

		return false;
	}

	void UBSAnalyseFrame() {
		PROFILE("UBS Analyse Frame");
		// if not record time, reset
		if (!mUBSTimerInitialized) {
			UBSResetAndBeginSearch();
			mUBSTimerInitialized = true;
			mUBSLastTime		 = CpuTimer::getCurrentTimePoint();
			return;
		}

		// get delta time [in seconds]
		CpuTimer::TimePoint currentTime = CpuTimer::getCurrentTimePoint();
		float deltaTime					= CpuTimer::calcDuration(mUBSLastTime, currentTime) * 1e-3;
		mUBSLastTime					= currentTime;

		const auto frameSize	  = getFrameSize();
		const auto resolutionSize = frameSize[0] * frameSize[1];

		// in fact, we don't desire the first state to be optimal[all fixed RR]
		// so we can deal with the first one causally
		// now add 1 more attempt/spp for the first one

		// get error for last frame
		if (mUBSLastError < 0) {
			// first attempt
			// reset state, but keep lastState = 0 (initial state)
			int tmp = mUBSLastSearchState;
			UBSResetAndBeginSearch();
			mUBSLastSearchState = tmp;
		} else {
			if (mUBSLastSearchState >= 0) {
				// calculate the efficiency
				// [TODO] ignore constant

				// in real app, we do not calc error, so we should substract the time
				// RTX 3080, time = 0.17ms
				mUBSLastDeltaTime -= 0.17f * 1e-3;
				float efficiency = resolutionSize / (mUBSLastDeltaTime * mUBSLastError);
#if 0
			Log(Info, "State: %s, Error: %f, Time: %f(%.2f), Efficiency: %f",
				UBSState2String(mUBSLastSearchState).c_str(), mUBSLastError / resolutionSize,
				mUBSLastDeltaTime, 1.0f / mUBSLastDeltaTime, efficiency);
#endif
				UBSRecordEfficiency(efficiency);
			}
		}

		mUBSLastDeltaTime = deltaTime;
		const auto stream = gpContext->cudaStream;

		// calculate error for current frame
		mUBSLastError = -1.0f;

		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				RGB ref			   = mRenderedImage->getPixel(tid).head<3>();
				Spectrum LSpectrum = mPixelState->mL[tid] / mSpp;
				RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

				// relative error, avoid emphasizing the error in the dark region
				float err				= ((L - ref).square() / (ref.square() + 1e-2f)).mean();
				mUBSErrorBufferGPU[tid] = err;
			},
			gpContext->cudaStream);

		// calculate error sum
		const float *rrsArray = mRRSArray;
		float *sum			  = mTempGPUBuffer;
		float *sumPos		  = sum + mUBSErrorSumPosInGPUBuffer;
		float *partSum		  = sum + (32 * 4); // warp size * 4

		calcSum2PassAsync<false>(mUBSErrorBufferGPU, sumPos, partSum, resolutionSize, stream);
		cudaMemcpyAsync(&mUBSLastError, sumPos, sizeof(float), cudaMemcpyDeviceToHost, stream);
		// cudaMemcpy(&mUBSLastError, sumPos, sizeof(float), cudaMemcpyDeviceToHost);

		return;
	}

	// must check mUseNNCache before using
	void resetNN(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples);
	void resizeNN(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples);
	void updateNNCache();

	void controlExp();

	friend void to_json(json &j, const EARSPathTracer &p) {
		j.update({{"spp_pre_iter", p.mSppPerIteration},
				  {"random_offset", p.mRandomOffset},
				  {"octree_max_memory", p.mOctreeMaxMemory},
				  {"auto_build_octree", p.mAutoBuild},
				  {"enable_rrs_after_pre_training", p.mEnableRRSAfterPreTraining},
				  {"outlier_number", p.mTopK},
				  {"pre_training_iter", p.mPreTrainingIterations},
				  {"training_iter", p.mTrainingIterations},
				  {"rrs_clamp_min", p.RRS_CLAMP_MIN},
				  {"rrs_clamp_max", p.RRS_CLAMP_MAX},
				  {"rrs_normalize_rate", p.RRS_NORMALIZE_RATE},
				  {"nee", p.mEnableNEE},
				  {"max_depth", p.mMaxDepth},
				  {"rr", p.mProbRR},
				  {"enable_clamp", p.mEnableClamp},
				  {"clamp_max", p.mClampMax},
				  {"max_rate_for_path_nodes_buffer", p.mMaxRateForPathNodesBuffer}});
	}

	friend void from_json(const json &j, EARSPathTracer &p) {
		p.mRandomOffset				 = j.value("random_offset", 0);
		p.mSppPerIteration			 = j.value("spp_per_iter", 64);
		p.mOctreeMaxMemory			 = j.value("octree_max_memory", 24);
		p.mAutoBuild				 = j.value("auto_build_octree", false);
		p.mEnableRRSAfterPreTraining = j.value("enable_rrs_after_pre_training", false);
		p.mTopK						 = j.value("outlier_number", 10);
		p.mPreTrainingIterations	 = j.value("pre_training_iter", 3);
		p.mTwiceIterInterval		 = j.value("twice_iter_interval", 8u);
		p.mTrainingIterations		 = j.value("training_iter", -1);
		p.RRS_CLAMP_MIN				 = j.value("rrs_clamp_min", 0.5f);
		p.RRS_CLAMP_MAX				 = j.value("rrs_clamp_max", 5.0f);
		p.RRS_NORMALIZE_RATE		 = j.value("rrs_normalize_rate", 0.5f);
		p.mRRSNormalizeUseCeil		 = j.value("rrs_normalize_use_ceil", false);
		p.mEnableNEE				 = j.value("nee", true);
		p.mMaxDepth					 = j.value("max_depth", 10);
		p.mProbRR					 = j.value("rr", 1.0);
		p.mEnableClamp				 = j.value("enable_clamp", false);
		p.mClampMax					 = j.value("clamp_max", 1e3f);
		p.mMaxRateForPathNodesBuffer = j.value("max_rate_for_path_nodes_buffer", 4);
		p.mKeepLastIterStatistics	 = j.value("keep_last_iter_statistics", false);
		p.mBatchPerFrame			 = j.value("batch_per_frame", 5);

		p.mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer =
			j.value("use_acc_buffer_not_denoised_ears", false);
		p.mDonnotNormalizeWhenLessThanOne = j.value("donnot_normalize_when_less_than_one", true);

		// adrrsn, earsn, adrrs, ears
		std::string methods = j.value("rrs_method", "ears");
		p.mRRSMethod		= methods[0] == 'a' ? RRSKind::ADRRS : RRSKind::EARS;
		p.mExpUseUBS		= methods == "ears+";

		// ears nn cache
		int method_length = (int) methods.length();
		assert(method_length >= 4); // shortest: ears
		char lastChar		  = methods.back();
		p.mUseNNCache		  = (lastChar == 'n' || lastChar == 't') ? true : false; // earsn, earst
		p.mUseNNCacheWithTree = (lastChar == 't') ? true : false;					 // earst
		p.mUseNNCacheWithTree &= p.mUseNNCache;

		Log(Info, "Methods: %s, Use NN Cache: %s", methods.c_str(),
			p.mUseNNCache ? "true" : "false");

		if (p.mUseNNCache) {
			if (j.contains("nrrs_config")) {
				string config_path = j.at("nrrs_config");
				std::ifstream f(config_path);
				if (f.fail()) {
					Log(Fatal, "Open network config file failed!");
				}
				p.mConfig = json::parse(f, nullptr, true, true)["nn"];
			} else {
				Log(Fatal, "Network config do not specified!");
			}
		}

		p.mDenoiseTask = std::make_shared<DenoiseTask>();
		p.mDenoiseTask->from_json(j.value("denoise_params", json{}));

		// Experiment
		// from global config
		const auto globalConfig = gpContext->globalConfig;

		p.mExpOn			= globalConfig.value("exp_on", false);
		p.mExpTrainTime		= globalConfig.value("exp_train_time", 60.0f);
		p.mExpInferenceTime = globalConfig.value("exp_inference_time", 60.0f);
		p.mExpOutputFile	= globalConfig.value("exp_output_file", "result.exr");
		p.mExpState			= 0;
	}
};

KRR_ENUM_DEFINE(EARSPathTracer::RRSKind, {
											 {EARSPathTracer::RRSKind::ADRRS, "ADRRS"},
											 {EARSPathTracer::RRSKind::EARS, "EARS"},
										 })
NAMESPACE_END(krr)