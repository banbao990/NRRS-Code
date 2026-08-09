#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/async/reduce.h>

#include "render/common/commoncudautilshost.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include "device/gpustd.h"
#include "render/common/commoncudautils.h"
#include "render/common/commoncudautilshost.h"
#include "nrrsparams.h"
#include "integrator.h"
#include "render/profiler/profiler.h"
#include "util/ema.h"
#include "util/film.h"

#include "train.h"

NAMESPACE_BEGIN(krr)
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////////////////////// pure simple render process /////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void NRRSPathTracer::changeRenderMode() {
	auto &param		  = mNRRSParams;
	Allocator &alloc  = *gpContext->alloc;
	const auto stream = gpContext->cudaStream;
	cudaStreamSynchronize(stream);
	if (param.mSimpleRenderOn) {
		// delete normal items, keep the pointers, it is small
		for (int i = 0; i < 2; ++i) {
			if (mRayQueue[i]) {
				// alloc.delete_object(mRayQueue[i]);
				// mRayQueue[i] = nullptr;
				mRayQueue[i]->resize(0, alloc);
			}
		}
		if (mMissRayQueue) {
			mMissRayQueue->resize(0, alloc);
		}
		if (mHitLightRayQueue) {
			mHitLightRayQueue->resize(0, alloc);
		}
		if (mShadowRayQueue) {
			mShadowRayQueue->resize(0, alloc);
		}
		if (mScatterRayQueue) {
			mScatterRayQueue->resize(0, alloc);
		}
	} else {
		// delete simple items
		for (int i = 0; i < 2; ++i) {
			if (mRayQueueSimple[i]) {
				mRayQueueSimple[i]->resize(0, alloc);
			}
		}
		if (mMissRayQueueSimple) {
			mMissRayQueueSimple->resize(0, alloc);
		}
		if (mHitLightRayQueueSimple) {
			mHitLightRayQueueSimple->resize(0, alloc);
		}
		if (mShadowRayQueueSimple) {
			mShadowRayQueueSimple->resize(0, alloc);
		}
		if (mScatterRayQueueSimple) {
			mScatterRayQueueSimple->resize(0, alloc);
		}
	}
	initialize(true);
}

void NRRSPathTracer::renderSimple(RenderContext *context) {
	PROFILE("Simple Mode");

	const auto frameSize	  = getFrameSize();
	const auto stream		  = gpContext->cudaStream;
	const uint resolutionSize = frameSize[0] * frameSize[1];

	for (int sampleId = 0; sampleId < mSamplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueueSimple(0)->reset(); }, stream);
		generateCameraRaysSimple(sampleId);

		// [STEP#2] do mRadiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(
				KRR_DEVICE_LAMBDA() {
					nextRayQueueSimple(depth)->reset();
					mHitLightRayQueueSimple->reset();
					mMissRayQueueSimple->reset();
					mShadowRayQueueSimple->reset();
					mScatterRayQueueSimple->reset();
					mScatterTidQueue->reset();
					mInferenceQueue->reset();
				},
				stream);

			const bool shouldInfer = isInferMode(depth);

			// [STEP#2.1] find closest intersections, filling in mScatterRayQueue and hitLightQueue
			traceClosestSimple(depth);

			if (shouldInfer) {
				// get inferenceQueueSize
				uint *p =
					(uint *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mInferQueueSizeInCPUBuffer);
				*p = 0;
				cudaMemcpyAsync(p, ((byte *) mInferenceQueue) + mInferenceQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToHost, stream);

				// specular queue size
				p = (uint *) (mNRRSParams.mTempGPUBuffer + mNRRSParams.mSpecularTidSizeInGPUBuffer);
				cudaMemcpyAsync(p, ((byte *) mScatterTidQueue) + mScatterTidQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToDevice, stream);
			}

			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleEmissiveHitSimple();
			handleMissSimple();

			// [Sidestory] break on maximum bounce, but after handling emissive intersections.
			if (depth == mMaxDepth) {
				break;
			}

			// -> [STEP#2.25] generate RRS Number
			if (shouldInfer) {
				generateRRSNumber<true, false>(depth);
			}

			// [STEP#2.3] handle intersections and shadow rays

			// tracing shadow rays should be before increment depth (the NEE contribution not
			// included in current vertex)
			handleIntersectionsSimple(depth);

			if (mEnableNEE) {
				traceShadowSimple();
			}
		}
	}

	// write results of the current frame...
	writeResultToRenderTarget<true>(context);
}

void NRRSPathTracer::generateCameraRaysSimple(int sampleId) {
	PROFILE("Generate camera rays(Simple)");
	SWPTRayQueue *cameraRayQueue = currentRayQueueSimple(0);
	auto frameSize				 = getFrameSize();
	const uint resolutionSize	 = frameSize[0] * frameSize[1];

	GPUParallelFor(
		resolutionSize,
		KRR_DEVICE_LAMBDA(const int pixelId) {
			// pixelID = tid
			Sampler sampler		= &mPixelState->mSampler[pixelId];
			Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
			Ray cameraRay		= mCamera->getRay(pixelCoord, frameSize, sampler);
			cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		},
		gpContext->cudaStream);
}

void NRRSPathTracer::traceClosestSimple(const int depth) {
	PROFILE("Trace intersect rays(Simple)");
	static LaunchParameters<NRRSPathTracer> params = {};

	params.mTraversable		 = mBackend->getRootTraversable();
	params.mSceneData		 = mBackend->getSceneData();
	params.mCurrentRayQueue	 = (NRRSRayQueue *) currentRayQueueSimple(depth);
	params.mMissRayQueue	 = (NRRSMissRayQueue *) mMissRayQueueSimple;
	params.mHitLightRayQueue = (NRRSHitLightRayQueue *) mHitLightRayQueueSimple;
	params.mScatterRayQueue	 = (NRRSScatterRayQueue *) mScatterRayQueueSimple;
	params.mNextRayQueue	 = (NRRSRayQueue *) nextRayQueueSimple(depth);
	params.mInferenceQueue	 = mInferenceQueue;
	params.mScatterTidQueue	 = mScatterTidQueue;
	// params.mRenderedImage	 = mRenderedImage;

	mBackend->launch(params, "ClosestSimple", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mCurrentRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += ((SWPTRayQueue *) rayQueue)->size(); },
			gpContext->cudaStream);
	}
}

void NRRSPathTracer::handleEmissiveHitSimple() {
	PROFILE("Process intersected rays(Simple)");
	ForAllQueued(
		mHitLightRayQueueSimple, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const SWPTHitLightWorkItem &w) {
			// RGB: ignore lambda
			Spectrum Le		= w.mLight.L(w.mPos, w.mNormal, w.mUv, w.mWo, {});
			float misWeight = 1;
			if (mEnableNEE && w.mDepth && !(w.mBsdfType & BSDF_SPECULAR)) {
				Light light = w.mLight;
				Interaction intr(w.mPos, w.mWo, w.mNormal, w.mUv);
				float lightPdf = light.pdfLi(intr, w.mCtx) * mLightSampler.pdf(light);
				misWeight	   = evalMIS(w.mPdf, lightPdf);
			}
			Spectrum contrib = Le * w.mThp * misWeight;
			mPixelState->addRadianceAtomic(w.mPixelId, contrib);
		},
		gpContext->cudaStream);
}

void NRRSPathTracer::handleMissSimple() {
	PROFILE("Process escaped rays(Simple)");
	const rt::SceneData &mSceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(
		mMissRayQueueSimple, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const SWPTMissRayWorkItem &w) {
			Spectrum L = {};
			Interaction intr(w.mRay.origin);
			for (const rt::InfiniteLight &light : mSceneData.infiniteLights) {
				float misWeight = 1;
				if (mEnableNEE && w.mDepth && !(w.mBsdfType & BSDF_SPECULAR)) {
					float lightPdf = light.pdfLi(intr, w.mCtx) * mLightSampler.pdf(&light);
					misWeight	   = evalMIS(w.mPdf, lightPdf);
				}
				// RGB: ignore lambda
				L += light.Li(w.mRay.dir, {}) * misWeight;
			}
			Spectrum contrib = L * w.mThp;
			mPixelState->addRadianceAtomic(w.mPixelId, contrib);
		},
		gpContext->cudaStream);
}

void NRRSPathTracer::handleIntersectionsSimple(const int depth) {
	PROFILE("Process intersections(Simple)");
	const auto stream	   = gpContext->cudaStream;
	const bool shouldInfer = isInferMode(depth);

	if (!shouldInfer) {
		ForAllQueued(
			mScatterRayQueueSimple, mMaxQueueSize,
			KRR_DEVICE_LAMBDA(SWPTScatterRayWorkItem & w) {
				// TODO: maybe sort will make higher coherence
				// maybe move to Closest hit shader is also ok
				const uint tid = blockIdx.x * blockDim.x + threadIdx.x;

				Sampler sampler = &mPixelState->mSampler[tid];
				// decide whether to terminate here (Russian Roulette).
				if (sampler.get1D() >= mFixedProbRR) {
					return;
				}
				w.mThp /= mFixedProbRR;

				generateScatteredRaysSimple(w, sampler, depth);
			},
			stream);
	} else {
		GPUParallelFor(
			mMaxQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				if (tid == 0) {
					if (mScatterTidQueue->size() > mMaxQueueSize) {
						printf("depth = %d, scatterTidQueue->size() > maxQueueSize, %d > %d\n",
							   depth, mScatterTidQueue->size(), mMaxQueueSize);
					}
				}
				if (tid >= mScatterTidQueue->size()) {
					return;
				}
				const uint sid = mScatterTidQueue->mTid[tid];

				SWPTScatterRayWorkItem sitem = mScatterRayQueueSimple->operator[](sid); // copy
				Sampler sampler				 = &mPixelState->mSampler[tid];
				generateScatteredRaysSimple(sitem, sampler, depth);
			},
			stream);
	}
}

KRR_DEVICE_FUNCTION void
NRRSPathTracer::generateScatteredRaysSimple(const SWPTScatterRayWorkItem &w, Sampler &sampler,
											const int depth) {

	const SurfaceInteraction &intr = w.mIntr;
	Vector3f woLocal			   = intr.toLocal(intr.wo);
	BSDFType bsdfType			   = intr.getBsdfType();

	/* Sampling direct illumination (L_d) with NEE */
	if (mEnableNEE && (bsdfType & BSDF_SMOOTH)) {
		SampledLight sampledLight = mLightSampler.sample(sampler.get1D());
		Light light				  = sampledLight.light;
		// RGB: ignore lambda
		LightSample ls	 = light.sampleLi(sampler.get2D(), {intr.p, intr.n}, {});
		Ray shadowRay	 = intr.spawnRayTo(ls.intr);
		Vector3f wiWorld = normalize(shadowRay.dir);
		Vector3f wiLocal = intr.toLocal(wiWorld);

		float lightPdf = sampledLight.pdf * ls.pdf;
		float misWeight{1};
		Spectrum bsdfVal = BxDF::f(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);

		float bsdfPdf = BxDF::pdf(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);
		misWeight	  = evalMIS(lightPdf, bsdfPdf);
		if (lightPdf > 0 && !isnan(misWeight) && !isinf(misWeight) && bsdfVal.any()) {
			SWPTShadowRayWorkItem sw = {};

			sw.mRay		= shadowRay;
			sw.mLi		= ls.L;
			sw.mPixelId = w.mPixelId;
			sw.mMaxT	= 1;
			sw.mDepth	= w.mDepth + 1;
			sw.mThp		= w.mThp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;

			if (sw.mThp.any()) {
				mShadowRayQueueSimple->push(sw);
			}
		}
	}

	/* sample BSDF */
	BSDFSample sample = BxDF::sample(intr, woLocal, sampler, (int) intr.sd.bsdfType);
	if (sample.pdf > 0 && sample.f.any()) {
		// check should get the L_i from network or not
		Vector3f wiWorld = intr.toWorld(sample.wi);
		Spectrum thp	 = w.mThp * sample.f * fabs(sample.wi[2]) / sample.pdf;

		if (any(thp)) {
			SWPTRayWorkItem r = {};

			Vector3f p	= offsetRayOrigin(intr.p, intr.n, wiWorld);
			r.mBsdfType = sample.flags;
			r.mPdf		= sample.pdf;
			r.mRay		= {p, wiWorld};
			r.mCtx		= {intr.p, intr.n};
			r.mPixelId	= w.mPixelId;
			r.mDepth	= w.mDepth + 1;
			r.mThp		= thp;
			nextRayQueueSimple(depth)->push(r);
		}
	}
}

void NRRSPathTracer::traceShadowSimple() {
	PROFILE("Trace shadow rays(Simple)");
	static LaunchParameters<NRRSPathTracer> params = {};

	params.mTraversable	   = mBackend->getRootTraversable();
	params.mSceneData	   = mBackend->getSceneData();
	params.mShadowRayQueue = (NRRSShadowRayQueue *) mShadowRayQueueSimple;
	params.mPixelState	   = mPixelState;

	mBackend->launch(params, "ShadowSimple", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mShadowRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += ((SWPTShadowRayQueue *) rayQueue)->size(); },
			gpContext->cudaStream);
	}
}

NAMESPACE_END(krr)