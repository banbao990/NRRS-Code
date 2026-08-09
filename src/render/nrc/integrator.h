#pragma once
#include "json.hpp"
#include "tiny-cuda-nn/common.h"

#include "camera.h"
#include "file.h"
#include "renderpass.h"
#include "scene.h"
#include "window.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/optix.h"
#include "nrcguided.h"
#include "nrc.h"
#include "nrctrain.h"
#include "workqueue.h"

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
using precision_t = tcnn::network_precision_t;

class Film;

class NRCPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<NRCPathTracer>;
	KRR_REGISTER_PASS_DEC(NRCPathTracer);

	NRCPathTracer() = default;
	NRCPathTracer(Scene &scene);
	~NRCPathTracer() = default;

	void resize(const Vector2i &size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(RenderContext *context) override;
	void endFrame(RenderContext *context) override;
	void render(RenderContext *context) override;
	void renderUI() override;
	void finalize() override;

	void initialize();

	string getName() const override { return "NRCPathTracer"; }

	template <bool TIsTraining, bool donNotTerminate> void handleIntersections(const int depth);
	template <bool TIsTraining> void handleEmissiveHit();
	template <bool TIsTraining> void handleMiss();
	void generateCameraRays(int sampleId);
	void traceClosest(int depth);
	void traceShadow(const bool isTraining);

	KRR_CALLABLE NRCRayQueue *mCurrentRayQueue(int depth) { return mRayQueue[depth & 1]; }
	KRR_CALLABLE NRCRayQueue *mNextRayQueue(int depth) { return mRayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

	// guided path routines
	void resetTraining();
	void trainStep();
	void renderTrainingSuffix();
	void generateCameraRaysTraining();

	void inferenceStep();
	void resetNetwork(json config);

	OptixBackend *mBackend;
	Camera::CameraData *mCamera{};
	LightSampler mLightSampler;

	// work queues
	NRCRayQueue *mRayQueue[2]{}; // switching bewteen current and next queue
	NRCMissRayQueue *mMissRayQueue{};
	NRCHitLightRayQueue *mHitLightRayQueue{};
	SWPTShadowRayQueue *mShadowRayQueue{};
	NRCScatterRayQueue *mScatterRayQueue{}; // bsdf evaluation (plus shadow ray generation)

	NRCInferenceQueue *mInferenceQueue{};

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
	float mProbRR{1};
	bool mEnableNEE{true};
	bool mEnableClamp{false};
	float mClampMax{1e4f};

	bool mOneStep{}, mTrainDebug{};
	bool mIsTrainingFinished{false};

	bool mDebugOn{false};
	int mDebugInt{0};

	// NRC parameters
	class NRCParams {
	public:
		KRR_HOST void beginFrame() {
			mTrainState.trainPixelOffset = mTrainState.trainPixelStride <= 1
											   ? 0
											   : mSampler.get1D() * mTrainState.trainPixelStride;
		}

		KRR_HOST void renderUI();

		KRR_CALLABLE bool isTrainingPixel(uint pixelId) const {
			return mTrainState.isTrainingPixel(pixelId);
		}

		NRCTrainState mTrainState;
		PCGSampler mSampler;
		uint mBatchPerFrame{5};
		uint mBatchSize{NRC_TRAIN_BATCH_SIZE};

		// NRC params
		float mStopC{0.01}; // hyperparameter

		bool mStopTraining{false};

		json mConfig;
		std::shared_ptr<tcnn::Network<float, precision_t>> mNetwork;
		std::shared_ptr<tcnn::Optimizer<precision_t>> mOptimizer;
		std::shared_ptr<tcnn::Loss<precision_t>> mLoss;
		std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> mTrainer;
	} mGuiding;

	friend void to_json(json &j, const NRCPathTracer &p) {
		j = json{{"nee", p.mEnableNEE},
				 {"max_depth", p.mMaxDepth},
				 {"rr", p.mProbRR},
				 {"enable_clamp", p.mEnableClamp},
				 {"clamp_max", p.mClampMax},
				 {"batch_per_frame", p.mGuiding.mBatchPerFrame},
				 {"batch_size", p.mGuiding.mBatchSize}};
	}

	friend void from_json(const json &j, NRCPathTracer &p) {
		p.mEnableNEE			  = j.value("nee", true);
		p.mMaxDepth				  = j.value("max_depth", 6);
		p.mProbRR				  = j.value("rr", 0.8);
		p.mEnableClamp			  = j.value("enable_clamp", false);
		p.mClampMax				  = j.value("clamp_max", 1e4f);
		p.mGuiding.mBatchPerFrame = j.value("batch_per_frame", 5);
		p.mGuiding.mBatchSize	  = j.value("batch_size", NRC_TRAIN_BATCH_SIZE);

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