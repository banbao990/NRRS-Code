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
#include "adnparameters.h"
#include "integrator.h"
#include "render/profiler/profiler.h"
#include "util/ema.h"
#include "util/film.h"

NAMESPACE_BEGIN(krr)
extern "C" char ADN_BB_PTX[];
using namespace tcnn;

namespace {
// training buffers / memories
tcnn::GPUMemory<float> sInferenceOutputBuffer;
tcnn::GPUMemory<float> sInferenceInputBuffer;

// lossgraph and training info logging / plotting
constexpr size_t LOSS_GRAPH_SIZE = 256;
std::vector<float> sLossGraph(LOSS_GRAPH_SIZE, 0);
size_t sNumLossSamples{0};
size_t sNumTrainingSamples{0};
Ema sCurLossScalar{Ema::Type::Time, 50};
} // namespace

template <bool tIsTraining> inline bool ADNPathTracer::isInferMode(const int depth) {
	return !tIsTraining && RRS_CLAMP_MIN < RRS_CLAMP_MAX; //&&(depth < NRC_MAX_TRAIN_DEPTH);
}

template <typename... Args>
KRR_DEVICE_FUNCTION void ADNPathTracer::debugPrint(uint pixelId, const char *fmt, Args &&...args) {
	if (pixelId == mDebugPixel) printf(fmt, std::forward<Args>(args)...);
}

void ADNPathTracer::initialize() {
	Allocator &alloc		  = *gpContext->alloc;
	const auto frameSize	  = getFrameSize();
	const uint resolutionSize = frameSize[0] * frameSize[1];
	mMaxQueueSize			  = 1.5f * resolutionSize;

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
	if (mInferenceQueue) {
		mInferenceQueue->resize(mMaxQueueSize, alloc);
	} else {
		mInferenceQueue = alloc.new_object<ADNInferenceQueue>(mMaxQueueSize, alloc);
	}
	if (mScatterTidQueue) {
		mScatterTidQueue->resize(mMaxQueueSize, alloc);
	} else {
		mScatterTidQueue = alloc.new_object<TidQueue>(mMaxQueueSize, alloc);
	}

	if (mPixelState) {
		mPixelState->resize(mMaxQueueSize, alloc);
	} else {
		mPixelState = alloc.new_object<SWPTPixelStateBuffer>(mMaxQueueSize, alloc);
	}

	if (mPathState) {
		mPathState->resize(resolutionSize, alloc);
	} else {
		mPathState = alloc.new_object<NRCPathPixelStateBuffer>(resolutionSize, alloc);
	}
	if (!mTrainBuffer) {
		mTrainBuffer = alloc.new_object<NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput>>(
			NRC_TRAIN_BUFFER_SIZE);
	}
	if (!mRenderedImage) {
		mRenderedImage = alloc.new_object<Film>(frameSize);
	} else {
		mRenderedImage->resize(frameSize); // will call reset()
	}

	cudaDeviceSynchronize();

	if (mGuiding.mRRSArray) {
		cudaFree(mGuiding.mRRSArray);
	}
	cudaMalloc(&mGuiding.mRRSArray, sizeof(float) * mMaxQueueSize);

	if (!mCamera) {
		cudaMalloc(&mCamera, sizeof(Camera::CameraData));
	}
	if (!mGuiding.mTempGPUBuffer) {
		cudaMalloc(&mGuiding.mTempGPUBuffer, sizeof(float) * mGuiding.mTempGPUBufferSize);
	}
	if (!mGuiding.mTempCPUBuffer) {
		mGuiding.mTempCPUBuffer = new float[mGuiding.mTempCPUBufferSize];
	}

	CUDA_SYNC_CHECK();
}

void ADNPathTracer::traceClosest(const int depth, const bool isTraining) {
	PROFILE("Trace intersect rays");
	static LaunchParameters<ADNPathTracer> params = {};

	params.mTraversable		 = mBackend->getRootTraversable();
	params.mSceneData		 = mBackend->getSceneData();
	params.mCurrentRayQueue	 = currentRayQueue(depth);
	params.mMissRayQueue	 = mMissRayQueue;
	params.mHitLightRayQueue = mHitLightRayQueue;
	params.mScatterRayQueue	 = mScatterRayQueue;
	params.mNextRayQueue	 = nextRayQueue(depth);
	params.mInferenceQueue	 = mInferenceQueue;
	params.mScatterTidQueue	 = mScatterTidQueue;
	params.mIsTraining		 = isTraining;
	params.mShowLi			 = mGuiding.mShowLi;
	params.mShowLiDepth		 = mGuiding.mShowLiDepth;

	mBackend->launch(params, "Closest", mMaxQueueSize, 1, 1);
}

void ADNPathTracer::traceShadow(const bool isTraining) {
	PROFILE("Trace shadow rays");
	static LaunchParameters<ADNPathTracer> params = {};

	params.mTraversable	   = mBackend->getRootTraversable();
	params.mSceneData	   = mBackend->getSceneData();
	params.mShadowRayQueue = mShadowRayQueue;
	params.mPixelState	   = mPixelState;
	params.mPathState	   = mPathState;
	params.mIsTraining	   = isTraining;
	mBackend->launch(params, "Shadow", mMaxQueueSize, 1, 1);
}

void ADNPathTracer::queryNetwork() {
	PROFILE("Query network");

	// just for show Li, so we don't need to optimize it, just sync
	const auto stream = gpContext->cudaStream;
	cudaStreamSynchronize(stream);
	const int inferenceQueueSize	 = mInferenceQueue->size();
	std::shared_ptr<Network> network = mGuiding.mNetwork;

	// prepare data
	LinearKernel(adn_generate_inference_data, stream, inferenceQueueSize, mInferenceQueue,
				 sInferenceInputBuffer.data(), mScene->getBoundingBox());
	// inference
	const int batchSizePad = next_multiple(inferenceQueueSize, (int) BATCH_SIZE_GRANULARITY);
	GPUMatrix<float> inferenceInputs(sInferenceInputBuffer.data(), NRC_DIM_INPUT, batchSizePad);
	GPUMatrix<float> inferenceOutputs(sInferenceOutputBuffer.data(), NRC_DIM_OUTPUT, batchSizePad);
	network->inference(stream, inferenceInputs, inferenceOutputs);

	// add mRadiance
	const float *inferData = sInferenceOutputBuffer.data();
	const bool grayScale   = mGuiding.mShowLiGrayScale;
	GPUParallelFor(
		inferenceQueueSize,
		KRR_DEVICE_LAMBDA(const int tid) {
			const uint sid	   = mInferenceQueue->mScatterQueueIndex[tid];
			const uint pixelId = mScatterRayQueue->mPixelId[sid];
			const Spectrum thp = mScatterRayQueue->mThp[sid];
			const RGB L		   = *((RGB *) (inferData + tid * NRC_DIM_OUTPUT));
			Spectrum LSpectrum = Spectrum::fromRGB(L, {}, {}, *KRR_DEFAULT_COLORSPACE_GPU);
			if (grayScale) {
				LSpectrum = Spectrum(LSpectrum.mean());
			}
			mPixelState->addRadianceAtomic(pixelId, LSpectrum * thp);
		},
		stream);
}

template <bool tIsTraining> void ADNPathTracer::handleEmissiveHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		mHitLightRayQueue, mMaxQueueSize,
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
			if (tIsTraining) {
				mPathState->recordRadiance(w.mPixelId, contrib);
				mPixelState->addRadiance(w.mPixelId, contrib);
			} else {
				mPixelState->addRadianceAtomic(w.mPixelId, contrib);
			}
		},
		gpContext->cudaStream);
}

template <bool tIsTraining> void ADNPathTracer::handleMiss() {
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
					float lightPdf = light.pdfLi(intr, w.mCtx) * mLightSampler.pdf(&light);
					misWeight	   = evalMIS(w.mPdf, lightPdf);
				}
				// RGB: ignore lambda
				L += light.Li(w.mRay.dir, {}) * misWeight;
			}
			Spectrum contrib = L * w.mThp;
			if (tIsTraining) {
				mPathState->recordRadiance(w.mPixelId, contrib);
				mPixelState->addRadiance(w.mPixelId, contrib);
			} else {
				mPixelState->addRadianceAtomic(w.mPixelId, contrib);
			}
		},
		gpContext->cudaStream);
}

template <bool tIsTraining> void ADNPathTracer::handleIntersections(const int depth) {
	PROFILE("Process intersections");
	const auto stream	   = gpContext->cudaStream;
	const bool shouldInfer = isInferMode<tIsTraining>(depth);
	if (!shouldInfer) {
		ForAllQueued(
			mScatterRayQueue, mMaxQueueSize,
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

				generateScatteredRays<tIsTraining>(w, sampler, depth);
			},
			stream);
	} else {
		GPUParallelFor(
			mMaxQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				if (tid == 0) {
					if (mScatterTidQueue->size() > mMaxQueueSize) {
						printf("scatterTidQueue->size() > maxQueueSize, %d > %d\n",
							   mScatterTidQueue->size(), mMaxQueueSize);
					}
				}
				if (tid >= mScatterTidQueue->size()) {
					return;
				}
				const uint sid = mScatterTidQueue->mTid[tid];

				SWPTScatterRayWorkItem sitem = mScatterRayQueue->operator[](sid); // copy
				Sampler sampler				 = &mPixelState->mSampler[tid];
				generateScatteredRays<tIsTraining>(sitem, sampler, depth);
			},
			stream);
	}
}

template <bool tIsTraining>
KRR_DEVICE_FUNCTION void ADNPathTracer::generateScatteredRays(const SWPTScatterRayWorkItem &w,
															  Sampler &sampler, const int depth) {

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
			sw.mThp		= w.mThp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;
			if (sw.mThp.any()) {
				mShadowRayQueue->push(sw);
			}
		}
	}

	/* sample BSDF */
	BSDFSample sample = BxDF::sample(intr, woLocal, sampler, (int) intr.sd.bsdfType);
	if (sample.pdf > 0 && sample.f.any()) {
		// check should get the L_i from network or not
		Vector3f wiWorld = intr.toWorld(sample.wi);
		Vector3f thp	 = w.mThp * sample.f * fabs(sample.wi[2]) / sample.pdf;

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
			nextRayQueue(depth)->push(r);

			if (tIsTraining) {
#ifndef USE_CAMERA_DIRECTION
				Spectrum thpTrain		= r.mThp;
				const Vector3f dirTrain = wiWorld;
#else
				Spectrum thpTrain		= w.mThp;
				const Vector3f dirTrain = intr.wo;
#endif
				Spectrum initRadiance = Spectrum::Zero();
				mPathState->incrementDepth(w.mPixelId, Ray{intr.p, dirTrain}, thpTrain,
										   initRadiance, sample.isDelta(), intr);
			}
		}
	}
}

void ADNPathTracer::generateCameraRays(int sampleId) {
	PROFILE("Generate camera rays");
	SWPTRayQueue *cameraRayQueue = currentRayQueue(0);
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

void ADNPathTracer::resize(const Vector2i &size) {
	if (size[0] * size[1] > NRC_MAX_RESOLUTION) {
		Log(Fatal, "Currently maximum number of pixels is limited to %d", NRC_MAX_RESOLUTION);
	}
	RenderPass::resize(size);
	initialize(); // need to resize the queues
}

void ADNPathTracer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	if (!mBackend) {
		mBackend	= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(ADN_BB_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false);
		mBackend->initialize(params);
	}
	mBackend->setScene(scene);
	mLightSampler = mBackend->getSceneData().lightSampler;
	initialize();
}

void ADNPathTracer::beginFrame(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}

	PROFILE("Begin frame");
	const auto stream = gpContext->cudaStream;
	auto frameSize	  = getFrameSize();
	cudaMemcpyAsync(mCamera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
					cudaMemcpyHostToDevice, stream);
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const int pixelId) {
			// reset per-pixel mRadiance & sample state
			Vector2i pixelCoord		 = {pixelId % frameSize[0], pixelId / frameSize[0]};
			mPixelState->mL[pixelId] = 0;
			mPixelState->mSampler[pixelId].setPixelSample(pixelCoord, mFrameId * mSamplesPerPixel);
			mPixelState->mSampler[pixelId].advance(256 * pixelId);
		},
		stream);
}

void ADNPathTracer::generateRRSNumber(const int depth) {
	PROFILE("RRS Number generation");
	const cudaStream_t &stream		 = gpContext->cudaStream;
	std::shared_ptr<Network> network = mGuiding.mNetwork;

	{
		PROFILE("Data preparation");
		LinearKernel(adn_generate_inference_data, stream, mMaxQueueSize, mInferenceQueue,
					 sInferenceInputBuffer.data(), mScene->getBoundingBox());
	}

	/////////////////// TODO: test
	// if (mDebugOn) {
	//	float *s   = mGuiding.mTempCPUBuffer;
	//	float sum1 = s[4];
	//	s += 32 * 4;
	//	cudaMemcpyAsync((void *) s, mGuiding.mTempGPUBuffer, sizeof(float) * 6,
	//					cudaMemcpyDeviceToHost, stream);
	//	cudaStreamSynchronize(stream);
	//	Log(Info, "sum1: %f, sum2: %f, sum3: %f, sum4: %f, sum[5]: %f", sum1, s[0], s[1], s[2],
	//		s[3], s[4]);
	// }
	/////////////////// TODO: test

	int inferenceQueueSize = 0;
	inferenceQueueSize = *(int *) (mGuiding.mTempCPUBuffer + mGuiding.mInferQueueSizeInCPUBuffer);
	if (!inferenceQueueSize) {
		// sync as this is allocated by cudaMallocManaged()
		cudaStreamSynchronize(stream);
		inferenceQueueSize = mInferenceQueue->size();
	}

	const int batchSizePad = next_multiple(inferenceQueueSize, (int) BATCH_SIZE_GRANULARITY);
	GPUMatrix<float> inferenceInputs(sInferenceInputBuffer.data(), NRC_DIM_INPUT, batchSizePad);
	GPUMatrix<float> inferenceOutputs(sInferenceOutputBuffer.data(), NRC_DIM_OUTPUT, batchSizePad);

	{
		PROFILE("Network inference");
		network->inference(stream, inferenceInputs, inferenceOutputs);
	}

	{
		PROFILE("Gen RRS Number");
		const float *inferData	   = sInferenceOutputBuffer.data();
		float *rrsArray			   = mGuiding.mRRSArray;
		const bool useWeightWindow = mGuiding.mUseWeightWindow;

		GPUParallelFor(
			inferenceQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				const uint sid	   = mInferenceQueue->mScatterQueueIndex[tid];
				const uint pixelId = mScatterRayQueue->mPixelId[sid];
				const RGB thp = mScatterRayQueue->mThp[sid].toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
				const RGBA refI = mRenderedImage->getPixel(pixelId);
				const RGB L		= clamp(*((RGB *) (inferData + tid * NRC_DIM_OUTPUT)), 1e-4f, 1e4f);

				float rrs = (L * thp / RGB(refI)).mean();
				if (useWeightWindow) {
					rrs = RRSWeightWindow(rrs);
				}
				rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);

				rrsArray[tid] = rrs;
			},
			stream);
	}

	{
		PROFILE("RRS Number Nomarlization");
		float *rrsArray	 = mGuiding.mRRSArray;
		float *sum		 = mGuiding.mTempGPUBuffer;
		float *sumRcpPos = sum + mGuiding.mSumPosInGPUBuffer;
		float *partSum	 = sum + (32 * 4); // warp size * 4
		calcSum2PassAsync<true>(rrsArray, sumRcpPos, partSum, inferenceQueueSize, stream);
	}

	{
		PROFILE("Gen Tids");
		const float *rrsSumRcp	  = mGuiding.mTempGPUBuffer + mGuiding.mSumPosInGPUBuffer;
		float *rrsArray			  = mGuiding.mRRSArray;
		const uint resolutionSize = getFrameSize()[0] * getFrameSize()[1];
		const uint *specularSize =
			(uint *) (mGuiding.mTempGPUBuffer + mGuiding.mSpecularTidSizeInGPUBuffer);

		GPUParallelFor(
			inferenceQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				const uint stid = mInferenceQueue->mScatterQueueIndex[tid];
				const RGB thp = mScatterRayQueue->mThp[stid].toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

				const float rrs =
					clamp(rrsArray[tid] * (RRS_NORMALIZE_RATE * (resolutionSize - *specularSize) *
										   (*rrsSumRcp)),
						  RRS_CLAMP_MIN, RRS_CLAMP_MAX);

				mScatterRayQueue->mThp[stid] = thp / rrs;

				auto sampler	= &mPixelState->mSampler[tid];
				const float rnd = sampler->get1D();
				rrsArray[tid]	= rrs; // update rrs

				int s			  = int(rrs);
				const float sLeft = rrs - s;

#if 1
				if (s > 0) {
					mScatterTidQueue->push(stid, s);
				}
				if (rnd <= sLeft) {
					mScatterTidQueue->push(stid);
				}
#else // faster but worse?
				s += (rnd <= sLeft) ? 1 : 0;
				if (s > 0) {
					scatterTidQueue->push(stid, s);
				}
#endif
			},
			stream);

		if (mDebugOn) {
			cudaStreamSynchronize(stream);
			auto scatterTidSize = mScatterTidQueue->size();
			if (scatterTidSize > mMaxQueueSize) {
				Log(Fatal, "%d > %d", scatterTidSize, mMaxQueueSize);
			}
			Log(Info, "r = %d / %d = %.4f, infer: %d, r_max = %.4f", scatterTidSize, resolutionSize,
				1.0f * scatterTidSize / resolutionSize, inferenceQueueSize,
				1.0f * scatterTidSize / mMaxQueueSize);
		}
	}
}

template <bool tIsTraining> void ADNPathTracer::renderInternal(RenderContext *context) {
	const auto frameSize	  = getFrameSize();
	const auto stream		  = gpContext->cudaStream;
	const uint resolutionSize = frameSize[0] * frameSize[1];

	for (int sampleId = 0; sampleId < mSamplesPerPixel; sampleId++) {
		// [STEP#0] reset training environment
		GPUCall(
			KRR_DEVICE_LAMBDA() {
				currentRayQueue(0)->reset();
				if (tIsTraining) {
					mTrainBuffer->clear();
				}
			},
			stream);

		if (tIsTraining) {
			GPUParallelFor(
				resolutionSize,
				KRR_DEVICE_LAMBDA(const int pixelId) {
					Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
					mPathState->reset(pixelId);
				},
				stream);
		}

		// [STEP#1] generate camera / primary rays
		generateCameraRays(sampleId);

		// [STEP#2] do mRadiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(
				KRR_DEVICE_LAMBDA() {
					nextRayQueue(depth)->reset();
					mHitLightRayQueue->reset();
					mMissRayQueue->reset();
					mShadowRayQueue->reset();
					mScatterRayQueue->reset();
					mScatterTidQueue->reset();
					mInferenceQueue->reset();
				},
				stream);

			const bool shouldInfer = isInferMode<tIsTraining>(depth);

			// [STEP#2.1] find closest intersections, filling in mScatterRayQueue and hitLightQueue
			traceClosest(depth, tIsTraining);

			if (shouldInfer) {
				// get inferenceQueueSize
				uint *p = (uint *) (mGuiding.mTempCPUBuffer + mGuiding.mInferQueueSizeInCPUBuffer);
				*p		= 0;
				cudaMemcpyAsync(p, ((byte *) mInferenceQueue) + mInferenceQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToHost, stream);

				// specular queue size
				p = (uint *) (mGuiding.mTempGPUBuffer + mGuiding.mSpecularTidSizeInGPUBuffer);
				cudaMemcpyAsync(p, ((byte *) mScatterTidQueue) + mScatterTidQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToDevice, stream);
			}

			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleEmissiveHit<tIsTraining>();
			handleMiss<tIsTraining>();

			const bool queryNetworkThenBreak =
				(mGuiding.mShowLi && (depth >= mGuiding.mShowLiDepth));
			if (queryNetworkThenBreak) {
				if (mGuiding.mShowLiAddRadiance) {
					queryNetwork();
				}
				break;
			}

			// [Sidestory] break on maximum bounce, but after handling emissive intersections.
			if (depth == mMaxDepth) {
				break;
			}

			// -> [STEP#2.25] generate RRS Number
			if (shouldInfer) {
				generateRRSNumber(depth);
			}

			// [STEP#2.3] handle intersections and shadow rays

			// tracing shadow rays should be before increment depth (the NEE contribution not
			// included in current vertex)
			handleIntersections<tIsTraining>(depth);
			if (mEnableNEE) {
				traceShadow(tIsTraining);
			}
		}

		// 1spp end

		if (tIsTraining) {
			trainStep();
		}
	}

	CudaRenderTarget cudaFrame = context->getColorTexture()->getCudaRenderTarget();

	// write results of the current frame...
	const bool showRenderedImage = mShowRenderedImage;
	const bool showLi			 = mGuiding.mShowLi;
	GPUParallelFor(
		resolutionSize,
		KRR_DEVICE_LAMBDA(const int pixelId) {
			Spectrum LSpectrum = mPixelState->mL[pixelId] / mSamplesPerPixel;
			RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
			if (mEnableClamp) {
				L = clamp(L, 0.f, mClampMax);
			}
			if (!showLi) {
				mRenderedImage->put(RGBA(L, 1), pixelId);
			}

			if (showRenderedImage) {
				cudaFrame.write(mRenderedImage->getPixel(pixelId), pixelId);
			} else {
				cudaFrame.write(RGBA(L, 1), pixelId);
			}
		},
		stream);
}

void ADNPathTracer::render(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}

	// update the mRenderedImage if the scene/camera changes
	if (mScene->getChanges()) {
		// camera changes will enter this branch
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
	}
	static size_t lastResetFrame = 0;
	auto lastSceneUpdates		 = mScene->getSceneGraph()->getLastUpdateRecord();
	if (lastSceneUpdates.updateFlags != SceneGraphNode::UpdateFlags::None &&
		lastResetFrame < lastSceneUpdates.frameIndex) {
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
		lastResetFrame = lastSceneUpdates.frameIndex;
	}

	PROFILE("ADN Path Tracer");

	if (mGuiding.mStopTraining) {
		renderInternal<false>(context);
	} else {
		renderInternal<true>(context);
	}
}

void ADNPathTracer::endFrame(RenderContext *context) { mFrameId++; }

void ADNPathTracer::finalize() {
	if (mCamera) {
		cudaFree(mCamera);
	}
	if (mGuiding.mRRSArray) {
		cudaFree(mGuiding.mRRSArray);
	}
	if (mGuiding.mTempGPUBuffer) {
		cudaFree(mGuiding.mTempGPUBuffer);
	}
	if (mGuiding.mTempCPUBuffer) {
		delete[] mGuiding.mTempCPUBuffer;
	}
}

void ADNPathTracer::ADNParams::renderUI(ADNPathTracer *pass) {
	ui::Checkbox("Stop Training", &mStopTraining);
	if (mStopTraining && !pass->mShowRenderedImage) {
		ui::Checkbox("Use Weight Window", &mUseWeightWindow);
		ui::Checkbox("Show Li", &mShowLi);
		if (mShowLi) {
			ui::Checkbox("Gray Scale", &mShowLiGrayScale);
			ui::Checkbox("Add Li Radiance", &mShowLiAddRadiance);
			ui::SliderInt("Min Depth to Query Network", &mShowLiDepth, 0, 10);
		}
	}
	if (!mStopTraining) {
		// TODO: guidedState can not hold on so many rays
		mShowLi = false;
	}

	static float lr = mOptimizer->learning_rate();
	if (ui::DragFloat("Learning rate", &lr, 1e-6f, 0, 1e-1, "%.6f")) {
		mOptimizer->set_learning_rate(lr);
	}
	if (ui::TreeNode("Advanced training options")) {
		if (ui::InputInt("Batch per frame", (int *) &mBatchPerFrame, 1, 1)) {
			mBatchPerFrame = max(0U, mBatchPerFrame);
		}
		// using rl2Loss				  = tcnn::RelativeL2LuminanceLoss<precision_t>;
		// std::shared_ptr<rl2Loss> loss = std::dynamic_pointer_cast<rl2Loss>(mLoss);
		// ui::Checkbox("loss clamp on", &(loss->mClampOn));
		// ui::SliderFloat("loss clamp max", &(loss->mClampMax), 0.0f, 1000.0f);
		ui::TreePop();
	}
	ui::Text("Current step: %d; %d samples; loss: %f", sNumLossSamples, sNumTrainingSamples,
			 sCurLossScalar.emaVal());
	ui::PlotLines("Loss graph", sLossGraph.data(), min(sNumLossSamples, sLossGraph.size()),
				  sNumLossSamples < LOSS_GRAPH_SIZE ? 0 : sNumLossSamples % LOSS_GRAPH_SIZE, 0,
				  FLT_MAX, FLT_MAX, ImVec2(0, 50));
}

void ADNPathTracer::renderUI() {
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &mSamplesPerPixel);
	ui::InputInt("Max bounces", &mMaxDepth, 1);
	ui::SliderFloat("Russian roulette", &mFixedProbRR, 0, 1);
	ui::Checkbox("Enable NEE", &mEnableNEE);

	ui::SliderFloat("RRS Max", &RRS_CLAMP_MAX, 1.0f, 20.0f);
	ui::SliderFloat("RRS Min", &RRS_CLAMP_MIN, 0.05f, 1.0f);
	ui::SliderFloat("RRS Normalize Rate", &RRS_NORMALIZE_RATE, 0.1f, 0.95f);
	ui::Checkbox("Show rendered image", &mShowRenderedImage);

	if (ui::Button("Reset rendered image")) {
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
	}

	if (mGuiding.mNetwork) {
		ui::Text("ADNParams");
		mGuiding.renderUI(this);
	}
	ui::Text("Debugging");
	ui::Checkbox("Debug On", &mDebugOn);
	ui::InputInt("DebugInt", &mDebugInt);
	ui::Checkbox("Debug output", &mDebugOutput);
	if (mDebugOutput) {
		ui::SameLine();
		ui::InputInt("Debug pixel:", (int *) &mDebugPixel);
	}
	if (ui::Button("Reset parameters")) {
		resetTraining();
	}
	ui::Checkbox("Clamping pixel value", &mEnableClamp);
	if (mEnableClamp) {
		ui::SameLine();
		ui::DragFloat("Max:", &mClampMax, 1, 1, 1e5f, "%.1f");
	}
}

void ADNPathTracer::resetNetwork(json config) {
	using namespace tcnn;
	tcnn::free_gpu_memory_arena(gpContext->cudaStream);

	mGuiding.mConfig	   = config;
	json &encoding_config  = config["encoding"];
	json &optimizer_config = config["optimizer"];
	json &network_config   = config["network"];
	json &loss_config	   = config["loss"];

	mGuiding.mOptimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
	mGuiding.mLoss.reset(tcnn::create_loss<precision_t>(loss_config));
	mGuiding.mNetwork = std::make_shared<NetworkWithInputEncoding>(NRC_DIM_INPUT, NRC_DIM_OUTPUT,
																   encoding_config, network_config);

	mGuiding.mTrainer = std::make_shared<Trainer>(mGuiding.mNetwork, mGuiding.mOptimizer,
												  mGuiding.mLoss, KRR_DEFAULT_RND_SEED);

	Log(Info, "Network has a padded output width of %d", mGuiding.mNetwork->padded_output_width());
	sInferenceInputBuffer  = GPUMemory<float>(NRC_DIM_INPUT * NRC_MAX_INFERENCE_NUM);
	sInferenceOutputBuffer = GPUMemory<float>(NRC_DIM_OUTPUT * NRC_MAX_INFERENCE_NUM);
	// TODO: NRC_MAX_INFERENCE_NUM: 720p, use maxResolution instead

	// mGuiding.trainer->initialize_params(); // already do in the Trainer's constructor
	mGuiding.mSampler.setSeed(KRR_DEFAULT_RND_SEED);
	CUDA_SYNC_CHECK();
}

void ADNPathTracer::resetTraining() {
	mGuiding.mTrainer->initialize_params();
	sNumLossSamples = 0;
}

void ADNPathTracer::trainStep() {
	PROFILE("Training");
	const cudaStream_t &stream		 = gpContext->cudaStream;
	const uint resolutionSize		 = getFrameSize()[0] * getFrameSize()[1];
	std::shared_ptr<Network> network = mGuiding.mNetwork;
	if (!network) {
		logFatal("Network not initialized!");
	}

	{
		PROFILE("Data preparation");
		LinearKernel(adn_generate_training_data, stream, resolutionSize, mTrainBuffer, mPathState,
					 mScene->getBoundingBox());
	}

	cudaStreamSynchronize(stream);
	sNumTrainingSamples	 = mTrainBuffer->size();
	const auto inputPos	 = mTrainBuffer->inputs();
	const auto outputPos = mTrainBuffer->outputs();

	uint numTrainBatches =
		min((uint) sNumTrainingSamples / mGuiding.mBatchSize + 1, mGuiding.mBatchPerFrame);

	float loss = 0.0f;

	for (int iter = 0; iter < numTrainBatches; iter++) {
		size_t localBatchSize =
			min(sNumTrainingSamples - iter * mGuiding.mBatchSize, (size_t) mGuiding.mBatchSize);
		localBatchSize -= localBatchSize % 128;
		if (localBatchSize < NRC_MIN_TRAIN_BATCH_SIZE) {
			break;
		}
		float *inputData			 = (float *) (inputPos + iter * mGuiding.mBatchSize);
		NRCNetworkOutput *outputData = outputPos + iter * mGuiding.mBatchSize;
		// drop the unaligned samples
		const int localBSPad = previous_multiple<int>(localBatchSize, BATCH_SIZE_GRANULARITY);
		GPUMatrix<float> networkInputs(inputData, NRC_DIM_INPUT, localBSPad);
		GPUMatrix<float> networkOutputs((float *) outputData, NRC_DIM_OUTPUT, localBSPad);
		{
			PROFILE("Train step");
			auto ctx = mGuiding.mTrainer->training_step(stream, networkInputs, networkOutputs);
			loss += mGuiding.mTrainer->loss(stream, *ctx);
		}
	}
	sCurLossScalar.update(loss / numTrainBatches);
	sLossGraph[sNumLossSamples++ % LOSS_GRAPH_SIZE] = sCurLossScalar.emaVal();
}

KRR_REGISTER_PASS_DEF(ADNPathTracer);
NAMESPACE_END(krr)