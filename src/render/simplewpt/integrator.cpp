#include <cuda.h>
#include <cuda_runtime.h>
#include "device/cuda.h"
#include "device/gpustd.h"

#include "integrator.h"
#include "simplewpt.h"
#include "render/profiler/profiler.h"
#include "workqueue.h"
#include "render/color.h"

NAMESPACE_BEGIN(krr)

extern "C" char SIMPLEWPT_PTX[];

template <typename... Args>
KRR_DEVICE_FUNCTION void SimpleWavefrontPathTracer::debugPrint(uint pixelId, const char *fmt,
															   Args &&...args) {
	if (pixelId == mDebugPixel) {
		printf(fmt, std::forward<Args>(args)...);
	}
}

void SimpleWavefrontPathTracer::initialize() {
	Allocator &alloc = *gpContext->alloc;
	auto frameSize	 = getFrameSize();
	mMaxQueueSize	 = frameSize[0] * frameSize[1];
	CUDA_SYNC_CHECK(); // necessary, preventing kernel accessing memories tobe free'ed...
	for (int i = 0; i < 2; i++) {
		if (mRayQueue[i]) {
			mRayQueue[i]->resize(mMaxQueueSize, alloc);
		} else {
			mRayQueue[i] = alloc.new_object<SWPTRayQueue>(mMaxQueueSize, alloc);
		}
	}
	if (mMissRayQueue) {
		mMissRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mMissRayQueue = alloc.new_object<SWPTMissRayQueue>(mMaxQueueSize, alloc);
	}
	if (mHitLightRayQueue) {
		mHitLightRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mHitLightRayQueue = alloc.new_object<SWPTHitLightRayQueue>(mMaxQueueSize, alloc);
	}
	if (mShadowRayQueue) {
		mShadowRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mShadowRayQueue = alloc.new_object<SWPTShadowRayQueue>(mMaxQueueSize, alloc);
	}
	if (mScatterRayQueue) {
		mScatterRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mScatterRayQueue = alloc.new_object<SWPTScatterRayQueue>(mMaxQueueSize, alloc);
	}
	if (mPixelState) {
		mPixelState->resize(mMaxQueueSize, alloc);
	} else {
		mPixelState = alloc.new_object<SWPTPixelStateBuffer>(mMaxQueueSize, alloc);
	}

	if (mExpOn) {
		if (mExpImage) {
			mExpImage->resize(frameSize);
		} else {
			mExpImage = alloc.new_object<Film>(frameSize);
		}
		mExpImage->reset();
		if (!mExpRayCounter) {
			cudaMalloc(&mExpRayCounter, sizeof(float));
			cudaMemset(mExpRayCounter, 0, sizeof(float));
		}
	}

	cudaDeviceSynchronize();
	if (!mCamera) {
		mCamera = alloc.new_object<Camera::CameraData>();
	}
	CUDA_SYNC_CHECK();
}

void SimpleWavefrontPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");
	static LaunchParameters<SimpleWavefrontPathTracer> params = {};

	params.mTraversable		 = mBackend->getRootTraversable();
	params.mSceneData		 = mBackend->getSceneData();
	params.mCurrentRayQueue	 = mCurrentRayQueue(depth);
	params.mMissRayQueue	 = mMissRayQueue;
	params.mHitLightRayQueue = mHitLightRayQueue;
	params.mScatterRayQueue	 = mScatterRayQueue;
	params.mNextRayQueue	 = mNextRayQueue(depth);
	mBackend->launch(params, "Closest", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mCurrentRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += rayQueue->size(); }, gpContext->cudaStream);
	}
}

void SimpleWavefrontPathTracer::traceShadow() {
	PROFILE("Trace shadow rays");
	static LaunchParameters<SimpleWavefrontPathTracer> params = {};

	params.mTraversable	   = mBackend->getRootTraversable();
	params.mSceneData	   = mBackend->getSceneData();
	params.mShadowRayQueue = mShadowRayQueue;
	params.mPixelState	   = mPixelState;
	mBackend->launch(params, "Shadow", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mShadowRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += rayQueue->size(); }, gpContext->cudaStream);
	}
}

void SimpleWavefrontPathTracer::handleHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		mHitLightRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const SWPTHitLightWorkItem &w) {
			// RGB: ignore lambda
			Spectrum Le		= w.mLight.L(w.mPos, w.mNormal, w.mUv, w.mWo, {});
			float misWeight = 1;
			// Simple understanding: if the sampled component is a delta func, then
			// it has infinite values and has 1 MIS weights.
			if (mEnableNEE && w.mDepth && !(w.mBsdfType & BSDF_SPECULAR)) {
				Light light = w.mLight;
				Interaction intr(w.mPos, w.mWo, w.mNormal, w.mUv);
				float lightPdf = light.pdfLi(intr, w.mCtx) * mLightSampler.pdf(light);
				float bsdfPdf  = w.mPdf;
				misWeight	   = evalMIS(bsdfPdf, lightPdf);
			}
			mPixelState->addRadiance(w.mPixelId, Le * w.mThp * misWeight);
		},
		gpContext->cudaStream);
}

void SimpleWavefrontPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	const rt::SceneData &mSceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(
		mMissRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const SWPTMissRayWorkItem &w) {
			Spectrum L = {};
			Interaction intr(w.mRay.origin);
			for (const rt::InfiniteLight &light : mSceneData.infiniteLights) {
				float misWeight = 1;
				if (mEnableNEE && w.mDepth && !(w.mBsdfType & BSDF_SPECULAR)) {
					float bsdfPdf  = w.mPdf;
					float lightPdf = light.pdfLi(intr, w.mCtx) * mLightSampler.pdf(&light);
					misWeight	   = evalMIS(bsdfPdf, lightPdf);
				}
				// RGB: ignore lambda
				L += light.Li(w.mRay.dir, {}) * misWeight;
			}
			mPixelState->addRadiance(w.mPixelId, w.mThp * L);
		},
		gpContext->cudaStream);
}

void SimpleWavefrontPathTracer::generateScatterRays(const int depth) {
	PROFILE("Generate scatter rays");
	ForAllQueued(
		mScatterRayQueue, mMaxQueueSize, KRR_DEVICE_LAMBDA(SWPTScatterRayWorkItem & w) {
			Sampler sampler = &mPixelState->mSampler[w.mPixelId];
			/*  Russian Roulette: If the path is terminated by this vertex,
				then NEE should not be evaluated */
			if (sampler.get1D() >= mProbRR) {
				return;
			}
			w.mThp /= mProbRR;
			const SurfaceInteraction &intr = w.mIntr;
			Vector3f woLocal			   = intr.toLocal(intr.wo);
			BSDFType bsdfType			   = intr.getBsdfType();
			/* sample direct lighting */
			if (mEnableNEE && (bsdfType & BSDF_SMOOTH)) {
				SampledLight sampledLight = mLightSampler.sample(sampler.get1D());
				Light light				  = sampledLight.light;
				// RGB: ignore lambda
				LightSample ls	 = light.sampleLi(sampler.get2D(), {intr.p, intr.n}, {});
				Ray shadowRay	 = intr.spawnRayTo(ls.intr);
				Vector3f wiWorld = normalize(shadowRay.dir);
				Vector3f wiLocal = intr.toLocal(wiWorld);

				float lightPdf = sampledLight.pdf * ls.pdf;
				float misWeight{1.0f};
				Spectrum bsdfVal = BxDF::f(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);
				float bsdfPdf	 = BxDF::pdf(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);
				misWeight		 = evalMIS(lightPdf, bsdfPdf);
				if (lightPdf > 0 && !isnan(misWeight) && !isinf(misWeight) && bsdfVal.any()) {
					SWPTShadowRayWorkItem sw = {};
					sw.mRay					 = shadowRay;
					sw.mLi					 = ls.L;
					sw.mPixelId				 = w.mPixelId;
					sw.mMaxT				 = 1;
					sw.mThp = w.mThp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;
					if (sw.mThp.any()) {
						mShadowRayQueue->push(sw);
					}
				}
			}

			/* sample BSDF */
			BSDFSample sample = BxDF::sample(intr, woLocal, sampler, (int) intr.sd.bsdfType);
			if (sample.pdf > 0 && sample.f.any()) {
				Vector3f wiWorld  = intr.toWorld(sample.wi);
				SWPTRayWorkItem r = {};
				Vector3f p		  = offsetRayOrigin(intr.p, intr.n, wiWorld);
				r.mBsdfType		  = sample.flags;
				r.mPdf			  = sample.pdf;
				r.mRay			  = {p, wiWorld};
				r.mCtx			  = {intr.p, intr.n};
				r.mPixelId		  = w.mPixelId;
				r.mDepth		  = w.mDepth + 1;
				r.mThp			  = w.mThp * sample.f * fabs(sample.wi[2]) / sample.pdf;
				if (any(r.mThp)) {
					mNextRayQueue(depth)->push(r);
				}
			}
		});
}

void SimpleWavefrontPathTracer::generateCameraRays(int sampleId) {
	PROFILE("Generate camera rays");
	SWPTRayQueue *cameraRayQueue = mCurrentRayQueue(0);
	auto frameSize				 = getFrameSize();
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(int pixelId) {
			Sampler sampler		= &mPixelState->mSampler[pixelId];
			Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
			Ray cameraRay		= mCamera->getRay(pixelCoord, frameSize, sampler);
			cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		},
		gpContext->cudaStream);
}

void SimpleWavefrontPathTracer::resize(const Vector2i &size) {
	RenderPass::resize(size);
	initialize(); // need to resize the queues
}

void SimpleWavefrontPathTracer::setScene(Scene::SharedPtr scene) {
	initialize();
	mScene = scene;
	if (!mBackend) {
		mBackend	= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(SIMPLEWPT_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false);
		mBackend->initialize(params);
	}
	mBackend->setScene(scene);
	mLightSampler = mBackend->getSceneData().lightSampler;
}

void SimpleWavefrontPathTracer::beginFrame(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}
	PROFILE("Begin frame");
	cudaMemcpyAsync(mCamera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
					cudaMemcpyHostToDevice, 0);
	size_t frameIndex = getFrameIndex();
	auto frameSize	  = getFrameSize();
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(int pixelId) { // reset per-pixel sample state
			Vector2i pixelCoord		 = {pixelId % frameSize[0], pixelId / frameSize[0]};
			mPixelState->mL[pixelId] = 0;
			mPixelState->mSampler[pixelId].setPixelSample(pixelCoord,
														  frameIndex * mSamplesPerPixel);
			mPixelState->mSampler[pixelId].advance(256 * pixelId + mRandomOffset);
		},
		gpContext->cudaStream);
}

void SimpleWavefrontPathTracer::render(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}

	if (mExpOn) {
		if (mExpState == 0) {
			Log(Info, "[Exp] Start Inference for %g seconds", mExpInferenceTime);
			mExpState	  = 2;
			mExpStartTime = CpuTimer::getCurrentTimePoint();
		}
		if (CpuTimer::calcElapsedTime(mExpStartTime) * 1e-3 >= mExpInferenceTime) {
			// save result and exit
			cudaDeviceSynchronize();
			mExpImage->save(mExpOutputFile);
			gpContext->requestExit();

			// save ray counter
			float rayCount;
			cudaMemcpy(&rayCount, mExpRayCounter, sizeof(float), cudaMemcpyDeviceToHost);
			json j;
			j["rays"]		= rayCount;
			j["infer-time"] = mExpInferenceTime;
			std::ofstream o(mExpOutputFile + ".json");
			o << std::setw(4) << j << std::endl;
			o.close();
		}
	}

	PROFILE("Wavefront Path Tracer");
	CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();
	for (int sampleId = 0; sampleId < mSamplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(KRR_DEVICE_LAMBDA() { mCurrentRayQueue(0)->reset(); }, gpContext->cudaStream);
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(
				KRR_DEVICE_LAMBDA() {
					mNextRayQueue(depth)->reset();
					mHitLightRayQueue->reset();
					mMissRayQueue->reset();
					mShadowRayQueue->reset();
					mScatterRayQueue->reset();
				},
				gpContext->cudaStream);
			// [STEP#2.1] find closest intersections, filling in mScatterRayQueue and hitLightQueue
			traceClosest(depth);
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleHit();
			handleMiss();
			// Break on maximum depth, but incorprate contribution from emissive hits.
			if (depth == mMaxDepth) {
				break;
			}
			// [STEP#2.3] evaluate materials & bsdfs, and generate shadow rays
			generateScatterRays(depth);
			// [STEP#2.4] trace shadow rays (next event estimation)
			if (mEnableNEE) {
				traceShadow();
			}
		}
	}
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(int pixelId) {
			Spectrum LSpectrum = mPixelState->mL[pixelId] / float(mSamplesPerPixel);
			RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
			if (mEnableClamp) {
				L = clamp(L, 0.f, mClampMax);
			}
			if (mExpState == 2) {
				mExpImage->put(RGBA(L, 1), pixelId);
				L = mExpImage->getPixel(pixelId).head<3>();
			}
			frameBuffer.write(RGBA(L, 1), pixelId);
		},
		gpContext->cudaStream);
}

void SimpleWavefrontPathTracer::renderUI() {
	ui::Text("Frame index (seed): %zd", getFrameIndex());
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &mSamplesPerPixel, 1, 1);
	ui::InputInt("Max bounces", &mMaxDepth, 1);
	ui::SliderFloat("Russian roulette", &mProbRR, 0, 1);
	ui::Checkbox("Enable NEE", &mEnableNEE);
	ui::Text("Debugging");
	ui::Checkbox("Debug output", &mDebugOutput);
	if (mDebugOutput) {
		ui::InputInt("Debug pixel:", (int *) &mDebugPixel);
	}
	ui::Checkbox("Clamping pixel value", &mEnableClamp);
	if (mEnableClamp) {
		ui::DragFloat("Max:", &mClampMax, 1, 1, 1e5f, "%.1f");
	}
}

void SimpleWavefrontPathTracer::finalize() {
	if (mExpRayCounter) {
		cudaFree(mExpRayCounter);
	}
}

KRR_REGISTER_PASS_DEF(SimpleWavefrontPathTracer);

NAMESPACE_END(krr)