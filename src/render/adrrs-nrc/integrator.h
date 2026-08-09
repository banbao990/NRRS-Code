#pragma once
#include "json.hpp"
#include <tiny-cuda-nn/common.h>

#include "camera.h"
#include "file.h"
#include "renderpass.h"
#include "scene.h"
#include "window.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/optix.h"
#include "adn.h"
#include "adntrain.h"
#include "workqueue.h"
#include "device/atomic.h"

namespace tcnn {
template <typename T> class Loss;
template <typename T> class Optimizer;
template <typename T> class Encoding;
template <typename T> class GPUMemory;
template <typename T> class GPUMatrixDynamic;
template <typename T, typename PARAMS_T> class Network;
template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
template <uint32_t N_DIMS, uint32_t RANK, typename T> class TrainableBuffer;
} // namespace tcnn

NAMESPACE_BEGIN(krr)

using nlohmann::json;
using precision_t			   = tcnn::network_precision_t;
using Network				   = tcnn::Network<float, precision_t>;
using Optimizer				   = tcnn::Optimizer<precision_t>;
using Loss					   = tcnn::Loss<precision_t>;
using Trainer				   = tcnn::Trainer<float, precision_t, precision_t>;
using NetworkWithInputEncoding = tcnn::NetworkWithInputEncoding<precision_t>;

class Film;

class ADNPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<ADNPathTracer>;
	KRR_REGISTER_PASS_DEC(ADNPathTracer);

	ADNPathTracer() = default;
	ADNPathTracer(Scene &scene);
	~ADNPathTracer() = default;

	void resize(const Vector2i &size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(RenderContext *context) override;
	void endFrame(RenderContext *context) override;
	void render(RenderContext *context) override;
	void renderUI() override;
	void finalize() override;

	void initialize();

	string getName() const override { return "ADNPathTracer"; }

	template <bool tIsTraining> void renderInternal(RenderContext *context);
	template <bool tIsTraining> void handleIntersections(const int depth);
	template <bool tIsTraining>
	KRR_DEVICE_FUNCTION void generateScatteredRays(const SWPTScatterRayWorkItem &w,
												   Sampler &sampler, const int depth);
	template <bool tIsTraining> void handleEmissiveHit();
	template <bool tIsTraining> void handleMiss();
	void queryNetwork();
	void generateCameraRays(int sampleId);
	void traceClosest(const int depth, const bool isTraining);

	void traceShadow(const bool isTraining);

	template <bool tIsTraining> inline bool isInferMode(const int depth);

	// RRS
	void generateRRSNumber(const int depth);

	KRR_CALLABLE SWPTRayQueue *currentRayQueue(int depth) { return mRayQueue[depth & 1]; }
	KRR_CALLABLE SWPTRayQueue *nextRayQueue(int depth) { return mRayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

	// guided path routines
	void resetTraining();
	void trainStep();

	OptixBackend *mBackend;
	Camera::CameraData *mCamera{};
	LightSampler mLightSampler;

	// work queues
	SWPTRayQueue *mRayQueue[2]{}; // switching bewteen current and next queue
	SWPTMissRayQueue *mMissRayQueue{};
	SWPTHitLightRayQueue *mHitLightRayQueue{};
	SWPTShadowRayQueue *mShadowRayQueue{};
	SWPTScatterRayQueue *mScatterRayQueue{}; // bsdf evaluation (plus shadow ray generation)

	TidQueue *mScatterTidQueue{};
	ADNInferenceQueue *mInferenceQueue{};

	SWPTPixelStateBuffer *mPixelState;
	NRCPathPixelStateBuffer *mPathState;
	NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput> *mTrainBuffer;

	// global properties
	bool mDebugOutput{};
	uint mDebugPixel{};
	int mMaxQueueSize;
	int mFrameId{0};

	// path tracing parameters
	int mSamplesPerPixel{1};
	int mMaxDepth{6};
	float mFixedProbRR{1};
	bool mEnableNEE{true};
	bool mEnableClamp{false};
	float mClampMax{1e4f};

	Film *mRenderedImage{nullptr};
	bool mShowRenderedImage{false};

	// Debug
	bool mDebugOn{false};
	int mDebugInt{0};

	// TODO: change to constexpr
	static constexpr uint INFER_MAX_QUEUE_SIZE_RATE = 2;
	float RRS_CLAMP_MAX = 5.0f, RRS_CLAMP_MIN = 0.5f, RRS_NORMALIZE_RATE = 0.5f;

	// ADN parameters
	class ADNParams {
	public:
		KRR_HOST void renderUI(ADNPathTracer *pass);

		PCGSampler mSampler;
		uint mBatchPerFrame{5};
		uint mBatchSize{NRC_TRAIN_BATCH_SIZE};

		bool mStopTraining{false};
		float *mRRSArray{nullptr};
		float *mTempGPUBuffer{nullptr};
		constexpr static uint mTempGPUBufferSize = 2048; // [note] only the 32*4 bytes can be used
		float *mTempCPUBuffer{nullptr};
		constexpr static uint mTempCPUBufferSize = mTempGPUBufferSize;
		bool mShowLi{false};
		bool mShowLiAddRadiance{true};
		bool mShowLiGrayScale{false};
		int mShowLiDepth{0}; // query the network at this depth
		bool mUseWeightWindow = false;

		constexpr static uint mSumPosInGPUBuffer		  = 0;
		constexpr static uint mSpecularTidSizeInGPUBuffer = sizeof(uint) / sizeof(float);
		constexpr static uint mInferQueueSizeInCPUBuffer  = 0;

		json mConfig;
		std::shared_ptr<Network> mNetwork;
		std::shared_ptr<Optimizer> mOptimizer;
		std::shared_ptr<Loss> mLoss;
		std::shared_ptr<Trainer> mTrainer;
	} mGuiding;

	void resetNetwork(json config);

	friend void to_json(json &j, const ADNPathTracer &p) {
		j = json{{"nee", p.mEnableNEE},
				 {"max_depth", p.mMaxDepth},
				 {"rr", p.mFixedProbRR},
				 {"enable_clamp", p.mEnableClamp},
				 {"clamp_max", p.mClampMax},
				 {"batch_per_frame", p.mGuiding.mBatchPerFrame},
				 {"batch_size", p.mGuiding.mBatchSize},
				 {"rrs_clamp_min", p.RRS_CLAMP_MIN},
				 {"rrs_clamp_max", p.RRS_CLAMP_MAX},
				 {"rrs_normalize_rate", p.RRS_NORMALIZE_RATE}};
	}

	friend void from_json(const json &j, ADNPathTracer &p) {
		p.mEnableNEE			  = j.value("nee", true);
		p.mMaxDepth				  = j.value("max_depth", 6);
		p.mFixedProbRR			  = j.value("rr", 0.8);
		p.mEnableClamp			  = j.value("enable_clamp", false);
		p.mClampMax				  = j.value("clamp_max", 1e4f);
		p.mGuiding.mBatchPerFrame = j.value("batch_per_frame", 5);
		p.mGuiding.mBatchSize	  = j.value("batch_size", NRC_TRAIN_BATCH_SIZE);
		p.RRS_CLAMP_MIN			  = j.value("rrs_clamp_min", 0.5f);
		p.RRS_CLAMP_MAX			  = j.value("rrs_clamp_max", 5.0f);
		p.RRS_NORMALIZE_RATE	  = j.value("rrs_normalize_rate", 0.5f);

		if (j.contains("config")) {
			string config_path = j.at("config");
			std::ifstream f(config_path);
			if (f.fail()) {
				logFatal("Open network config file failed!");
			}
			json config = json::parse(f, nullptr, true, true);
			p.resetNetwork(config["nn"]);
		} else {
			Log(Warning, "Network config do not specified! Assuming guiding is disabled.");
		}
	}
};

NAMESPACE_END(krr)