#pragma once
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"
#include "renderpass.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/optix.h"
#include "device/cuda.h"
#include "workqueue.h"
#include "util/film.h"
#include "device/timer.h"

NAMESPACE_BEGIN(krr)

class SimpleWavefrontPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<SimpleWavefrontPathTracer>;
	KRR_REGISTER_PASS_DEC(SimpleWavefrontPathTracer);

	SimpleWavefrontPathTracer()	 = default;
	~SimpleWavefrontPathTracer() = default;

	void resize(const Vector2i &size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(RenderContext *context) override;
	void render(RenderContext *context) override;
	void renderUI() override;
	void finalize() override;

	void initialize();

	string getName() const override { return "SimpleWavefrontPathTracer"; }

	void handleHit();
	void handleMiss();
	void generateScatterRays(const int depth);
	void generateCameraRays(int sampleId);
	void traceClosest(int depth);
	void traceShadow();

	KRR_CALLABLE SWPTRayQueue *mCurrentRayQueue(int depth) { return mRayQueue[depth & 1]; }
	KRR_CALLABLE SWPTRayQueue *mNextRayQueue(int depth) { return mRayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

	OptixBackend *mBackend{};
	Camera::CameraData *mCamera{};
	LightSampler mLightSampler;

	// work queues
	SWPTRayQueue *mRayQueue[2]{}; // switching bewteen current and next queue
	SWPTMissRayQueue *mMissRayQueue{};
	SWPTHitLightRayQueue *mHitLightRayQueue{};
	SWPTShadowRayQueue *mShadowRayQueue{};
	SWPTScatterRayQueue *mScatterRayQueue{};
	SWPTPixelStateBuffer *mPixelState;

	// path tracing parameters
	int mMaxQueueSize;
	int mSamplesPerPixel{1};
	int mMaxDepth{10};
	float mProbRR{0.8};
	bool mEnableNEE{};
	bool mDebugOutput{};
	bool mEnableClamp{false};
	uint mDebugPixel{};
	float mClampMax{1e3f};

	int mRandomOffset{0}; // used to offset the random seed for each pixel

	// Experiments
	bool mExpOn{false};
	float *mExpRayCounter{nullptr};
	float mExpInferenceTime{60.0f};
	std::string mExpOutputFile{};
	Film *mExpImage{nullptr};
	CpuTimer::TimePoint mExpStartTime{};
	CpuTimer::TimePoint mExpCurrentTime{};
	int mExpState{0};

	friend void to_json(json &j, const SimpleWavefrontPathTracer &p) {
		j = json{{"nee", p.mEnableNEE},
				 {"max_depth", p.mMaxDepth},
				 {"rr", p.mProbRR},
				 {"enable_clamp", p.mEnableClamp},
				 {"random_offset", p.mRandomOffset},
				 {"clamp_max", p.mClampMax}};
	}

	friend void from_json(const json &j, SimpleWavefrontPathTracer &p) {
		p.mEnableNEE	= j.value("nee", true);
		p.mMaxDepth		= j.value("max_depth", 10);
		p.mProbRR		= j.value("rr", 0.8);
		p.mEnableClamp	= j.value("enable_clamp", false);
		p.mClampMax		= j.value("clamp_max", 1e3f);
		p.mRandomOffset = j.value("random_offset", 0);

		// from global config
		const auto globalConfig = gpContext->globalConfig;

		p.mExpOn			= globalConfig.value("exp_on", false);
		p.mExpInferenceTime = globalConfig.value("exp_inference_time", 60.0f);
		p.mExpOutputFile	= globalConfig.value("exp_output_file", "result.exr");
		p.mExpState			= 0;
	}
};

NAMESPACE_END(krr)