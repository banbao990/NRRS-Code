#include "common.h"
#include "window.h"
#include "file.h"
#include "device/gpustd.h"

#include "util/check.h"
#include "util/film.h"

#include "ears.h"
#include "render/common/tree/tree.h"
#include "integrator.h"
#include "render/profiler/profiler.h"
#include "render/common/commoncudautils.h"
#include "render/common/commoncudautilshost.h"
#include "render/common/topk/top_k_gpu.h"
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/async/sort.h>

#include "util/ema.h"

NAMESPACE_BEGIN(krr)
extern "C" char EARS_PTX[];

namespace {
static size_t sGuidingTrainedFrames		= 0;
static size_t sTrainFramesThisIteration = 0;
static bool sTrainOneStep				= false;
const static char *sRRSMethods[]		= {"ADRRS", "EARS"};

tcnn::GPUMemory<float> sInferenceInputBuffer;
tcnn::GPUMemory<float> sInferenceOutputBuffer;

// lossgraph and training info logging / plotting
constexpr size_t LOSS_GRAPH_SIZE = 256;
std::vector<float> sLossGraph(LOSS_GRAPH_SIZE, 0);
size_t sNumLossSamples{0};
size_t sNumTrainingSamples{0};
Ema sCurLossScalar{Ema::Type::Time, 50};

} // namespace

template <typename... Args>
KRR_DEVICE void EARSPathTracer::debugPrint(const char *fmt, Args &&...args) {
	if (mDebugOn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid == mDebugIntNumber) {
			printf(fmt, std::forward<Args>(args)...);
		}
	}
}

template <typename... Args>
KRR_DEVICE void EARSPathTracer::debugPrintOneCall(const char *fmt, Args &&...args) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (mDebugOn && tid == 0) {
		printf(fmt, std::forward<Args>(args)...);
	}
}

void EARSPathTracer::resize(const Vector2i &size) {
	RenderPass::resize(size);
	initialize(); // need to resize the queues
	mDenoiseTask->resize(size);

	if (mUseNNCache) {
		const int resolutionSize = size[0] * size[1];
		resizeNN(size, mMaxQueueSize, resolutionSize * mMaxRateForPathNodesBuffer);
	}
}

void EARSPathTracer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	mDenoiseTask->setScene(scene);
	if (!mBackend) {
		mBackend	= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(EARS_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false);

		mBackend->initialize(params);
	}
	mBackend->setScene(scene);
	mLightSampler	 = mBackend->getSceneData().lightSampler;
	AABB aabb		 = scene->getBoundingBox();
	Allocator &alloc = *gpContext->alloc;

	if (mOctree) {
		delete mOctree;
	}
	const uint octreeMaxSize = mOctreeMaxMemory * 1024 * 1024;
	// when useNNCacheWithTree, cache cost in sampling node
	mOctree = alloc.new_object<Octree>(octreeMaxSize, mKeepLastIterStatistics, mUseNNCacheWithTree);
	mOctree->initialize(aabb, mOctreeInitDepth);

	CUDA_SYNC_CHECK();
}

void EARSPathTracer::initialize() {
	const auto stream = gpContext->cudaStream;

	Allocator &alloc		  = *gpContext->alloc;
	const auto frameSize	  = getFrameSize();
	const uint resolutionSize = frameSize[0] * frameSize[1];
	const uint tMaxQueueSize  = /*1.5f **/ resolutionSize;
	if (mMaxQueueSize == tMaxQueueSize) {
		return;
	}
	mMaxQueueSize = tMaxQueueSize;

	// We need this since CUDA 12 seems to have reduced the default stack size,
	// However, SD-Tree has some recursive routines that may exceed that size;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));

	// necessary, preventing kernel accessing memories tobe free'ed...
	CUDA_SYNC_CHECK();

	for (int i = 0; i < 2; i++) {
		if (mRayQueue[i]) {
			mRayQueue[i]->resize(mMaxQueueSize, alloc);
		} else {
			mRayQueue[i] = alloc.new_object<EARSRayQueue>(mMaxQueueSize, alloc);
		}
	}
	if (mMissRayQueue) {
		mMissRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mMissRayQueue = alloc.new_object<EARSMissRayQueue>(mMaxQueueSize, alloc);
	}
	if (mHitLightRayQueue) {
		mHitLightRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mHitLightRayQueue = alloc.new_object<EARSHitLightRayQueue>(mMaxQueueSize, alloc);
	}
	if (mShadowRayQueue) {
		mShadowRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mShadowRayQueue = alloc.new_object<EARSShadowRayQueue>(mMaxQueueSize, alloc);
	}
	if (mScatterRayQueue) {
		mScatterRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mScatterRayQueue = alloc.new_object<EARSScatterRayQueue>(mMaxQueueSize, alloc);
	}
	if (mPixelState) {
		mPixelState->resize(mMaxQueueSize, alloc);
	} else {
		mPixelState = alloc.new_object<EARSPixelStateBuffer>(mMaxQueueSize, alloc);
	}

	// mEnableLearning = !mEnableRRS
	if (mPathState) {
		mPathState->resize(resolutionSize * mMaxRateForPathNodesBuffer, alloc);
	} else {
		mPathState = alloc.new_object<EARSPathNodesBuffer>(
			resolutionSize * mMaxRateForPathNodesBuffer, alloc);
	}

	if (mScatterTidQueue) {
		mScatterTidQueue->resize(mMaxQueueSize, alloc);
	} else {
		mScatterTidQueue = alloc.new_object<TidQueue>(mMaxQueueSize, alloc);
	}
	if (mNonSpecularTidQueue) {
		mNonSpecularTidQueue->resize(mMaxQueueSize, alloc);
	} else {
		mNonSpecularTidQueue = alloc.new_object<EARSInferenceQueue>(mMaxQueueSize, alloc);
	}

	if (mUseNNCache) {
		const int maxTrainingSamples = resolutionSize * mMaxRateForPathNodesBuffer;
		if (mTrainBuffer) {
			mTrainBuffer->resize(maxTrainingSamples);
		} else {
			mTrainBuffer =
				alloc.new_object<NetworkTrainBuffer<EARSNetworkInput, EARSNetworkOutput>>(
					maxTrainingSamples);
		}
	}

	if (mShowRRSBuffer) {
		cudaFree(mShowRRSBuffer);
	}
	cudaMalloc(&mShowRRSBuffer, sizeof(float) * resolutionSize);

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

	if (mRenderedImage) {
		mRenderedImage->resize(frameSize);
	} else {
		mRenderedImage = alloc.new_object<Film>(frameSize);
	}
	mRenderedImage->reset();
	mRenderedImageSpp = 0u;
	if (mTempResolutionSizeBuffer) {
		cudaFree(mTempResolutionSizeBuffer);
	}
	cudaMalloc(&mTempResolutionSizeBuffer, sizeof(float) * resolutionSize);

	if (mRRSArray) {
		cudaFree(mRRSArray);
	}
	cudaMalloc(&mRRSArray, sizeof(float) * mMaxQueueSize);
	if (mRRSArrayCeil) {
		cudaFree(mRRSArrayCeil);
	}
	cudaMalloc(&mRRSArrayCeil, sizeof(float) * mMaxQueueSize);

	if (!mCamera) {
		cudaMalloc(&mCamera, sizeof(Camera::CameraData));
	}
	if (!mTempGPUBuffer) {
		cudaMalloc(&mTempGPUBuffer, sizeof(float) * mTempGPUBufferSize);
	}
	if (!mTempCPUBuffer) {
		mTempCPUBuffer = new float[mTempCPUBufferSize];
		FloatPointerWarpper *fp =
			(FloatPointerWarpper *) (mTempCPUBuffer + mEARSCostGPUPointerInCPUBuffer);
		fp->mData = mPixelState->mDepth;
	}

	if (!mImageStatistic) {
		cudaMalloc(&mImageStatistic, sizeof(EARSImageStatistic));
	}

	{
		// as the renderer design, pass->initialize() only called once out side itself
		static bool mDenoiseTaskInitialized = false;
		if (!mDenoiseTaskInitialized) {
			mDenoiseTask->initialize(frameSize[0], frameSize[1]);
			mDenoiseTaskInitialized = true;
		}
	}

	if (!mDebugStats) {
		cudaMalloc(&mDebugStats, sizeof(StatMinMaxAvgGPU));
	}
	if (!mUBSArray) {
		mUBSArray = new int[UBS_MAX_DEPTH];
		memset(mUBSArray, 0, sizeof(int) * UBS_MAX_DEPTH);
	}
	if (!mUBSArrayChar) {
		mUBSArrayChar = new char[UBS_MAX_DEPTH];
		memset(mUBSArrayChar, 0, sizeof(char) * UBS_MAX_DEPTH);
	}
	if (mUBSErrorBufferGPU) {
		cudaFree(mUBSErrorBufferGPU);
	}
	cudaMalloc(&mUBSErrorBufferGPU, sizeof(float) * resolutionSize);

	GPUCall(KRR_DEVICE_LAMBDA() { mImageStatistic->reset(); });

	CUDA_SYNC_CHECK();
}

void EARSPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");
	static LaunchParameters<EARSPathTracer> params = {};

	params.mTraversable		 = mBackend->getRootTraversable();
	params.mSceneData		 = mBackend->getSceneData();
	params.mCurrentRayQueue	 = mCurrentRayQueue(depth);
	params.mMissRayQueue	 = mMissRayQueue;
	params.mHitLightRayQueue = mHitLightRayQueue;
	params.mScatterRayQueue	 = mScatterRayQueue;
	params.mNextRayQueue	 = mNextRayQueue(depth);

	params.mEnableTraining		= mEnableLearning;
	params.mEnableRRS			= mEnableRRS;
	params.mNonSpecularTidQueue = mNonSpecularTidQueue;
	params.mScatterTidQueue		= mScatterTidQueue;

	params.mQueryNNCache = mUseNNCache && mShowLi && (depth >= mShowLiDepth);

	mBackend->launch(params, "Closest", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mCurrentRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += rayQueue->size(); }, gpContext->cudaStream);
	}
}

void EARSPathTracer::traceShadow() {
	PROFILE("Trace shadow rays");
	static LaunchParameters<EARSPathTracer> params = {};

	params.mTraversable	   = mBackend->getRootTraversable();
	params.mSceneData	   = mBackend->getSceneData();
	params.mShadowRayQueue = mShadowRayQueue;
	params.mPixelState	   = mPixelState;
	params.mPathState	   = mPathState;
	params.mEnableTraining = mEnableLearning;
	mBackend->launch(params, "Shadow", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mShadowRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += rayQueue->size(); }, gpContext->cudaStream);
	}
}

void EARSPathTracer::handleHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		mHitLightRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const EARSHitLightWorkItem &w) {
			// RGB: ignore lambda
			Spectrum Le		= w.mLight.L(w.mPos, w.mNormal, w.mUv, w.mWo, {});
			float misWeight = 1;
			if (mEnableNEE && w.mDepth && !(w.mBsdfType & BSDF_SPECULAR)) {
				Light light = w.mLight;
				Interaction intr(w.mPos, w.mWo, w.mNormal, w.mUv);
				float lightPdf = light.pdfLi(intr, w.mCtx) * mLightSampler.pdf(light);
				float bsdfPdf  = w.mPdf;
				misWeight	   = evalMIS(bsdfPdf, lightPdf);
			}
			const Spectrum contrib = Le * w.mThp * misWeight;
			mPixelState->addStatisticAtomic(w.mPixelId, w.mDepth, contrib);
			if (mEnableLearning) {
				mPathState->recordRadiance(w.mNodeIdx, contrib, false); // EARS
			}
		},
		gpContext->cudaStream);
}

void EARSPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	const rt::SceneData &mSceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(
		mMissRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const EARSMissRayWorkItem &w) {
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
			const Spectrum contrib = L * w.mThp;
			mPixelState->addStatisticAtomic(w.mPixelId, w.mDepth, contrib);
			if (mEnableLearning) {
				mPathState->recordRadiance(w.mNodeIdx, contrib, false); // EARS
			}
		},
		gpContext->cudaStream);
}

void EARSPathTracer::updateOctree() {
	PROFILE("Update Octree");
	const auto stream = gpContext->cudaStream;

	const auto frameSize	 = getFrameSize();
	const int resolutionSize = frameSize[0] * frameSize[1];

	GPUParallelFor(
		resolutionSize,
		KRR_DEVICE_LAMBDA(const int tid) {
			const int dataLength = mPathState->size();

			int itemIndex = tid;

			while (itemIndex < dataLength) {
				const EARSRadianceRecordItem w = (*mPathState)[itemIndex];

				// if (w.mNumSamples == 0) {
				// return;
				//}

				Octree::TrainingNode *trainingNode = nullptr;
				Octree::SamplingNode *samplingNode = nullptr;
				Octree::Node *trainingPosNode	   = nullptr;
				mOctree->lookup(w.mPos, w.mDir, samplingNode, trainingNode, trainingPosNode);

				// same as nrrs
				RGB L = RGB::Zero();
				for (int ch = 0; ch < RGB::dim; ch++) {
					if (w.mThp[ch] > M_EPSILON) {
						L[ch] = w.mL[ch] / w.mThp[ch];
					}
				}
				trainingNode->splatLrEstimate(L, w.mCost, 1u, trainingPosNode);

				itemIndex += resolutionSize; // next item
			}
		},
		stream);
	// mOctree->debug(nullptr);
}

void EARSPathTracer::queryRadiance() {
	PROFILE("Query Cache");

	const auto stream = gpContext->cudaStream;

	if (mUseNNCache && mShowLi_NNCache) {
		uint nonSpecularQueueSize = *(uint *) (mTempCPUBuffer + mNonSpecularQueueSizeInCPUBuffer);
		if (!nonSpecularQueueSize) {
			// sync as this is allocated by cudaMallocManaged()
			Log(Info, "[Sync][queryRadiance()]");
			cudaStreamSynchronize(stream);
			nonSpecularQueueSize = mNonSpecularTidQueue->size();
		}

		// [1] prepare inference input
		{
			float *networkInputPtr = sInferenceInputBuffer.data();
			const AABB sceneAABB   = mScene->getBoundingBox();
			GPUParallelFor(
				nonSpecularQueueSize,
				KRR_DEVICE_LAMBDA(const int tid) {
					EARSInferenceItem item = (*mNonSpecularTidQueue)[tid];

					EARSNetworkInput *input =
						(EARSNetworkInput *) (networkInputPtr + tid * LL2NET_DIM_INPUT);

					input->mPos		  = normalizeSpatialCoord(item.mPos, sceneAABB);
					input->mDir		  = utils::worldToLatLong(item.mDir); // 3d - > 2d
					input->mRoughness = warp_roughness(item.mRoughness);
				},
				stream);
		}

		// [2] inference
		{
			using namespace tcnn;
			const int batchSizePad = next_multiple(nonSpecularQueueSize, BATCH_SIZE_GRANULARITY);
			const GPUMatrix<float> inferenceInput(sInferenceInputBuffer.data(), LL2NET_DIM_INPUT,
												  batchSizePad);
			GPUMatrix<float> inferenceOutput(sInferenceOutputBuffer.data(), LL2NET_DIM_OUTPUT,
											 batchSizePad);
			mNetwork->inference(stream, inferenceInput, inferenceOutput);
		}

		// [3] update radiance
		const float *output = sInferenceOutputBuffer.data();
		GPUParallelFor(
			nonSpecularQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				uint sid	 = mNonSpecularTidQueue->mTid[tid];
				uint pixelId = mScatterRayQueue->mPixelId[sid];
				Spectrum thp = mScatterRayQueue->mThp[sid];

				// TODO: more types(L, L2, cost)
				const float *ptrL = (float *) (output + tid * LL2NET_DIM_OUTPUT);

				RGB L = *(RGB *) ptrL; // L

				if (mShowLiType == 1) {
					L = RGB(L.mean()); // L.mean()
				} else if (mShowLiType == 2) {
					RGB varL = *(RGB *) (ptrL + 3); // Var
					L		 = varL + L * L;		// L^2: E(X^2) = Var + (EX)^2
				} else if (mShowLiType == 3) {
					L	= RGB(*(ptrL + 6) * mShowLiCostWeightCoeff); // cost
					thp = Spectrum(1.0f);
				}

				mPixelState->addRadianceAtomic(
					pixelId, thp * Spectrum::fromRGB(L, {}, {}, *KRR_DEFAULT_COLORSPACE_GPU));
			},
			stream);

	} else {
		ForAllQueued(
			mScatterRayQueue, mMaxQueueSize,
			KRR_DEVICE_LAMBDA(const EARSScatterRayWorkItem &w) {
				// [note] sd.light != true
				const SurfaceInteraction &intr	   = w.mIntr;
				Octree::TrainingNode *trainingNode = nullptr;
				Octree::SamplingNode *samplingNode = nullptr;
				Octree::Node *trainingNodeLock	   = nullptr;
				if (mShowLiLookUpType == 1) {
					mOctree->lookupIgnoreInvalid(intr.p, intr.wo, samplingNode, trainingNode,
												 trainingNodeLock);
				} else if (mShowLiLookUpType == 2) {
					mOctree->lookupMaxDepth(intr.p, intr.wo, samplingNode, trainingNode,
											trainingNodeLock, mShowLiLookUpMaxDepth);
				} else {
					// 0 or default
					mOctree->lookup(intr.p, intr.wo, samplingNode, trainingNode, trainingNodeLock);
				}
				RGB L[2];
				Spectrum thp = w.mThp;
				if (mShowLiType == 0) {
					L[0] = trainingNode->getLrEstimate();
					L[1] = samplingNode->mLrEstimate;
				} else if (mShowLiType == 1) {
					L[0] = Spectrum(trainingNode->getLrEstimate().mean());
					L[1] = Spectrum(samplingNode->mLrEstimate.mean());
				} else if (mShowLiType == 2) {
					L[0] = trainingNode->getLrSecondMoment();
					L[1] = RGB(0);
				} else if (mShowLiType == 3) {
					if (mUseNNCacheWithTree) {
						float v = trainingNode->getLrCost();
						L[0]	= RGB(v);
						v		= samplingNode->mLrEstimate[1];
						L[1]	= RGB(v);
						thp		= Spectrum(1.0f); // w is a copy
					}
				}
				const uint idx = mShowLiSamplingNode ? 1 : 0;
				mPixelState->addRadianceAtomic(
					w.mPixelId,
					thp * Spectrum::fromRGB(L[idx], {}, {}, *KRR_DEFAULT_COLORSPACE_GPU));
			},
			stream);
	}
}

void EARSPathTracer::generateRRSNumber(const int depth) {
#define RECORD_RRS(type) (mShowRRS && (mShowRRSMode == type) && ((1 << depth) & mShowRRSWhichDepth))

	PROFILE("RRS Number generation");
	const cudaStream_t stream = gpContext->cudaStream;

	uint mScatterRayQueueSize = 0;
	mScatterRayQueueSize	  = *(uint *) (mTempCPUBuffer + mScatterRayQueueSizeInCPUBuffer);
	uint nonSpecularQueueSize = *(uint *) (mTempCPUBuffer + mNonSpecularQueueSizeInCPUBuffer);
	if (!mScatterRayQueueSize || !nonSpecularQueueSize) {
		// sync as this is allocated by cudaMallocManaged()
		Log(Info, "[Sync]");
		cudaStreamSynchronize(stream);
		mScatterRayQueueSize = mScatterRayQueue->size();
		nonSpecularQueueSize = mNonSpecularTidQueue->size();
	}
	const uint specularSize = mScatterRayQueueSize - nonSpecularQueueSize;

	{
		if (mDebugOn) {
			GPUCall(KRR_DEVICE_LAMBDA() { mDebugStats->init(); }, stream);
		}

		PROFILE("Gen RRS Number");
		RGB *denoisedBuffer = (RGB *) mDenoiseTask->getBuffer(DenoiseTask::DenoisedColor);

		const bool updateShowRRSBuffer0 = RECORD_RRS(BeforeClamp);
		const bool updateShowRRSBuffer1 = RECORD_RRS(AfterClamp);

		float *rrsArrayCeil = mRRSArrayCeil;
		const bool useCeil	= mRRSNormalizeUseCeil;

		if (mUseNNCache) {
			// [1] prepare inference input
			{
				float *networkInputPtr = sInferenceInputBuffer.data();
				const AABB sceneAABB   = mScene->getBoundingBox();
				GPUParallelFor(
					nonSpecularQueueSize,
					KRR_DEVICE_LAMBDA(const int tid) {
						EARSInferenceItem item = (*mNonSpecularTidQueue)[tid];

						EARSNetworkInput *input =
							(EARSNetworkInput *) (networkInputPtr + tid * LL2NET_DIM_INPUT);

						input->mPos		  = normalizeSpatialCoord(item.mPos, sceneAABB);
						input->mDir		  = utils::worldToLatLong(item.mDir); // 3d - > 2d
						input->mRoughness = warp_roughness(item.mRoughness);
					},
					stream);
			}
			// [2] nn inference
			{
				using namespace tcnn;
				const int batchSizePad =
					next_multiple(nonSpecularQueueSize, BATCH_SIZE_GRANULARITY);
				const GPUMatrix<float> inferenceInput(sInferenceInputBuffer.data(),
													  LL2NET_DIM_INPUT, batchSizePad);
				GPUMatrix<float> inferenceOutput(sInferenceOutputBuffer.data(), LL2NET_DIM_OUTPUT,
												 batchSizePad);
				mNetwork->inference(stream, inferenceInput, inferenceOutput);
			}
			// [3] gen original rrs
			{
				const float *output = sInferenceOutputBuffer.data();
				GPUParallelFor(
					nonSpecularQueueSize,
					KRR_DEVICE_LAMBDA(const int tid) {
						uint sid	 = mNonSpecularTidQueue->mTid[tid];
						uint pixelId = mScatterRayQueue->mPixelId[sid];

						RGB refI	  = mRenderedImage->getPixel(pixelId).head<3>(); // must valid
						const RGB thp = mScatterRayQueue->mThp[sid]
											.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU)
											.cwiseQuotient(refI + 1e-2f);

						const float *ptrL = output + tid * LL2NET_DIM_OUTPUT;

						const RGB L	   = RGB{ptrL[0], ptrL[1], ptrL[2]}; // L
						const RGB varL = RGB{ptrL[2], ptrL[3], ptrL[4]}; // var
						float cost	   = max(ptrL[6], 0.01f);			 // cost

						if (mUseNNCacheWithTree) {
							const Vector3f pos = mScatterRayQueue->mIntr[sid].soa->p[sid];
							const Vector3f wo  = mScatterRayQueue->mIntr[sid].soa->wo[sid];

							Octree::TrainingNode *trainingNode = nullptr;
							Octree::SamplingNode *samplingNode = nullptr;
							Octree::Node *trainingNodeLock	   = nullptr;
							mOctree->lookup(pos, wo, samplingNode, trainingNode, trainingNodeLock);
							cost = samplingNode->mLrEstimate[1];
						}

						float rrs = 1.0f;

						if (mRRSMethod == RRSKind::ADRRS) {
							rrs = (clamp(L, 1e-4f, 1e4f) * thp).mean();
						} else if (mRRSMethod == RRSKind::EARS) {

							RGB earsFactorR = ((varL + L * L) / cost).max(0); // 2nd-moment / cost
							RGB earsFactorS = (varL / cost).max(0);			  // variance / cost

							const RGB t2 = thp.cwiseProduct(thp);
							const float imageEarsFactor =
								mUseCurrentImageStatistic ? mImageStatistic->getEARSFactorCurrent()
														  : mImageStatistic->getEARSFactor();
							const float splittingFactorS =
								sqrt((t2 * earsFactorR).mean()) * imageEarsFactor;
							const float splittingFactorR =
								sqrt((t2 * earsFactorS).mean()) * imageEarsFactor;
							if (splittingFactorR > 1) {
								if (splittingFactorS < 1) {
									/// second moment and variance disagree on whether to split
									/// or RR, resort to doing nothing.
									rrs = 1.0f;
								} else {
									/// use variance only if both modes recommend splitting.
									rrs = splittingFactorS;
								}
							} else {
								/// use second moment only if it recommends RR.
								rrs = splittingFactorR;
							}

							if (isnan(rrs) || isinf(rrs)) {
								printf("[%d] rrs = %f, L = (%f, %f, %f), varL = (%f, %f, %f),"
									   "cost = %f(%f)"
									   " splittingFactorR = %f, splittingFactorS = %f,"
									   " imageEarsFactor = %f\n",
									   tid, rrs, L[0], L[1], L[2], varL[0], varL[1], varL[2], cost,
									   ptrL[6], splittingFactorR, splittingFactorS,
									   imageEarsFactor);
							}
							if (isnan(rrs)) {
								// in fact, won't come here
								printf("[EARS-NN] meet nan rrs!\n");
								rrs = 1.0f;
							}

							// debugPrint("[EARS-NN] tid = %d, r = %f, s = %f, rrs = %f\n", tid,
							// splittingFactorR, splittingFactorS, rrs);
						}

						const float rrsOrignal = rrs;

						rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);

						if (updateShowRRSBuffer0 || updateShowRRSBuffer1) {
							// very slow, we have got the same result before, however, it is for
							// debugging
							const float rrsRes = updateShowRRSBuffer0 ? rrsOrignal : rrs;
							// if just depth=0, there is no need to use atomicAdd
							// showRRSBuffer[pixelId] += rrsRes;
							atomicAdd(mShowRRSBuffer + pixelId, rrsRes);
						}

						mRRSArray[tid] = rrs;

						if (useCeil) {
							mRRSArrayCeil[tid] = ceilf(rrs);
						}
					},
					stream);
			}
		} else {

			// if UBS on, mLossUseNetworkOutputRRS has no effects
			const bool forceUseEARS =
				(mUBSSearch || mUseBestStrategy) ? UBSUseNetOutputRRS(depth) : false;

			GPUParallelFor(
				nonSpecularQueueSize, // real size
				KRR_DEVICE_LAMBDA(const int tid) {
					float rrs = 1.0f;

					const uint stid	   = mNonSpecularTidQueue->mTid[tid];
					const Vector3f pos = mScatterRayQueue->mIntr[stid].soa->p[stid];
					const Vector3f wo  = mScatterRayQueue->mIntr[stid].soa->wo[stid];
					const uint pixelId = mScatterRayQueue->mPixelId[stid];
					RGB refI;

					// must valid
					refI		  = mRenderedImage->getPixel(pixelId).head<3>();
					const RGB thp = mScatterRayQueue->mThp[stid]
										.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU)
										.cwiseQuotient(refI + 1e-2f);

					Octree::TrainingNode *trainingNode = nullptr;
					Octree::SamplingNode *samplingNode = nullptr;
					Octree::Node *trainingNodeLock	   = nullptr;
					mOctree->lookup(pos, wo, samplingNode, trainingNode, trainingNodeLock);

					if (!forceUseEARS && mRRSMethod == RRSKind::ADRRS) {
						RGB LFromOctree = samplingNode->mLrEstimate;
						const RGB L		= clamp(LFromOctree, 1e-4f, 1e4f);
						rrs				= (L * thp).mean();
					} else if (mRRSMethod == RRSKind::EARS) {
						const RGB t2				= thp.cwiseProduct(thp);
						const float imageEarsFactor = mUseCurrentImageStatistic
														  ? mImageStatistic->getEARSFactorCurrent()
														  : mImageStatistic->getEARSFactor();
						const float splittingFactorS =
							sqrt((t2 * samplingNode->mEarsFactorS).mean()) * imageEarsFactor;
						const float splittingFactorR =
							sqrt((t2 * samplingNode->mEarsFactorR).mean()) * imageEarsFactor;
						if (splittingFactorR > 1) {
							if (splittingFactorS < 1) {
								/// second moment and variance disagree on whether to split or
								/// RR, resort to doing nothing.
								rrs = 1.0f;
							} else {
								/// use variance only if both modes recommend splitting.
								rrs = splittingFactorS;
							}
						} else {
							/// use second moment only if it recommends RR.
							rrs = splittingFactorR;
						}

						// if (isnan(rrs) || isinf(rrs)) {
						//	printf("[%d] rrs = %f, splittingFactorR = %f, splittingFactorS = %f,
						//" 		   "imageEarsFactor = %f\n", 		   tid, rrs,
						// samplingNode->mEarsFactorS.mean(),
						// samplingNode->mEarsFactorR.mean(), imageEarsFactor);
						// }
						if (isnan(rrs)) {
							// in fact, won't come here
							printf("meet nan rrs!\n");
							rrs = 1.0f;
						}

						// debugPrint("[EARS] tid = %d, r = %f, s = %f, rrs = %f\n", tid,
						// splittingFactorR, splittingFactorS, rrs);

					} else {
						rrs = 1.0f; // all the same
					}

					// debugPrint("[RRS] tid = %d, rrs = %f\n", tid, rrs);
					// if (mDebugOn) {
					//	mDebugStats->record(rrs);
					// }

					const float rrsOrignal = rrs;

					rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);

					if (updateShowRRSBuffer0 || updateShowRRSBuffer1) {
						// very slow, we have got the same result before, however, it is for
						// debugging
						const float rrsRes = updateShowRRSBuffer0 ? rrsOrignal : rrs;
						// if just depth=0, there is no need to use atomicAdd
						// showRRSBuffer[pixelId] += rrsRes;
						atomicAdd(mShowRRSBuffer + pixelId, rrsRes);
					}

					mRRSArray[tid] = rrs;

					if (useCeil) {
						mRRSArrayCeil[tid] = ceilf(rrs);
					}
				},
				stream);
		}

		// if (mDebugOn) {
		//	GPUCall(
		//		KRR_DEVICE_LAMBDA() { mDebugStats->print(); }, stream);
		// }
	}

	{
		PROFILE("RRS Number Nomalization");
		float *sum		 = mTempGPUBuffer;
		float *rrsSumRcp = sum + mRRSSumPosInGPUBuffer;
		float *partSum	 = sum + mTempPosInGPUBuffer;

		const float *rrsArray = mRRSNormalizeUseCeil ? mRRSArrayCeil : mRRSArray;

		calcSum2PassAsync<true>(rrsArray, rrsSumRcp, partSum, nonSpecularQueueSize, stream);
	}

	{
		PROFILE("Gen Tids");
		const float *rrsSumRcp	  = mTempGPUBuffer + mRRSSumPosInGPUBuffer;
		float *rrsArray			  = mRRSArray;
		const uint resolutionSize = getFrameSize()[0] * getFrameSize()[1];

		const bool updateShowRRSBuffer2 = RECORD_RRS(Normalized);
		const bool updateShowRRSBuffer3 = RECORD_RRS(RealRay);

		const bool shouldScaleUp = !mDonnotNormalizeWhenLessThanOne;

		const bool useCeil = mRRSNormalizeUseCeil;

		GPUParallelFor(
			nonSpecularQueueSize, // real size
			KRR_DEVICE_LAMBDA(const int tid) {
				const uint stid	   = mNonSpecularTidQueue->mTid[tid];
				const Spectrum thp = mScatterRayQueue->mThp[stid];

				const float normalizeFactor = ((resolutionSize - specularSize) * (*rrsSumRcp)) *
											  (useCeil ? 1.0f : RRS_NORMALIZE_RATE);

				// if (tid == 0) {
				// printf("normalizeFactor: %f, rrsSum: %f\n", normalizeFactor, 1 / *rrsSumRcp);
				// }

				// float rrs = rrsArray[tid] * normalizeFactor;

				float rrs = rrsArray[tid];
				if (shouldScaleUp) {
					rrs = rrs * normalizeFactor;
				} else {
					// if sum < resolution, don't normalize
					rrs = rrs * (normalizeFactor < 1.0f ? normalizeFactor : 1.0f);
				}

				// rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);

				mScatterRayQueue->mThp[stid] = thp / rrs;

				auto sampler	= &mPixelState->mSampler[tid];
				const float rnd = sampler->get1D();
				// rrsArray[tid]	= rrs; // update rrs

				int s			  = int(rrs);
				const float sLeft = rrs - s;

				if (updateShowRRSBuffer2 || updateShowRRSBuffer3) {
					const uint pixelId = mScatterRayQueue->mPixelId[stid];
					float rrsRes	   = updateShowRRSBuffer2 ? rrs : float(s + (rnd <= sLeft));
					atomicAdd(mShowRRSBuffer + pixelId, rrsRes);
				}

				s += (rnd <= sLeft);
				if (s > 0) {
					mScatterTidQueue->push(stid, s);
				}
			},
			stream);
	}

#undef RECORD_RRS
}

template <bool tIsTraining>
KRR_DEVICE_FUNCTION void EARSPathTracer::generateScatteredRays(EARSScatterRayWorkItem &w,
															   Sampler &sampler, const int depth) {
	const SurfaceInteraction &intr = w.mIntr;
	BSDFType bsdfType			   = intr.getBsdfType();
	Vector3f woLocal			   = intr.toLocal(intr.wo);
	int nodeIdx					   = w.mNodeIdx;

	// get a node for current point, skip if specular
	if (tIsTraining && (bsdfType & BSDF_SMOOTH)) {
		// TODO: needs optimization, change sd.mWo from 3D to 2D
		// sd.mWo : view direction, normalized, last point -> current point
		// sd.pos: not normalized
		// mThp  : not multiplied by BSDF
		nodeIdx = mPathState->recordNode(w.mNodeIdx, intr.p, intr.wo, intr.sd.roughness, w.mThp);
		if (nodeIdx == -1) {
			nodeIdx = w.mNodeIdx;
		}
	}

	// sample light
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
		float bsdfPdf	 = BxDF::pdf(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);
		misWeight		 = evalMIS(lightPdf, bsdfPdf);
		if (lightPdf > 0 && !isnan(misWeight) && !isinf(misWeight) && bsdfVal.any()) {
			EARSShadowRayWorkItem sw = {};
			sw.mRay					 = shadowRay;
			sw.mLi					 = ls.L;
			sw.mThp					 = w.mThp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;
			sw.mPixelId				 = w.mPixelId;
			sw.mMaxT				 = 1;
			sw.mNodeIdx				 = nodeIdx;
			sw.mDepth				 = w.mDepth + 1;

			if (sw.mThp.any()) {
				mShadowRayQueue->push(sw);
			}
		}
	}

	// sample bsdf
	BSDFSample sample = BxDF::sample(intr, woLocal, sampler, (int) intr.sd.bsdfType);
	if (sample.pdf > 0 && sample.f.any()) {
		Vector3f wiWorld = intr.toWorld(sample.wi);
		Spectrum thp	 = w.mThp * sample.f * fabs(sample.wi[2]) / sample.pdf;

		if (any(thp)) {
			EARSRayWorkItem r = {};

			Vector3f p	= offsetRayOrigin(intr.p, intr.n, wiWorld);
			r.mBsdfType = sample.flags;
			r.mPdf		= sample.pdf;
			r.mRay		= {p, wiWorld};
			r.mCtx		= {intr.p, intr.n};
			r.mPixelId	= w.mPixelId;
			r.mDepth	= w.mDepth + 1;
			r.mThp		= thp;
			r.mNodeIdx	= nodeIdx;

			mNextRayQueue(depth)->push(r);
		}
	}
}

template <bool tIsTraining> void EARSPathTracer::handleIntersections(const int depth) {
	PROFILE("Handle intersections");
	const auto stream = gpContext->cudaStream;

	if (shouldGenRRS(depth)) {
		GPUParallelFor(
			mMaxQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				if (tid >= mScatterTidQueue->size()) {
					return;
				}

				if (tid == 0) {
					if (mScatterTidQueue->size() > mMaxQueueSize) {
						printf("mScatterTidQueue->size() > mMaxQueueSize, %d > %d\n",
							   mScatterTidQueue->size(), mMaxQueueSize);
					}
				}
				const uint sid = mScatterTidQueue->mTid[tid];

				EARSScatterRayWorkItem sitem = mScatterRayQueue->operator[](sid); // copy
				Sampler sampler				 = &mPixelState->mSampler[tid];
				generateScatteredRays<tIsTraining>(sitem, sampler, depth);
			},
			stream);
	} else {
		ForAllQueued(
			mScatterRayQueue, mMaxQueueSize,
			KRR_DEVICE_LAMBDA(EARSScatterRayWorkItem & w) {
				Sampler sampler = &mPixelState->mSampler[w.mPixelId];
				if (sampler.get1D() >= mProbRR) {
					return;
				}
				w.mThp /= mProbRR;
				generateScatteredRays<tIsTraining>(w, sampler, depth);
			},
			stream);
	}
}

void EARSPathTracer::generateCameraRays(const int sampleId) {
	PROFILE("Generate mCamera rays");
	EARSRayQueue *cameraRayQueue = mCurrentRayQueue(0);
	auto frameSize				 = getFrameSize();
	auto resolutionSize			 = frameSize[0] * frameSize[1];

	GPUParallelFor(
		resolutionSize,
		KRR_DEVICE_LAMBDA(int pixelId) {
			Sampler sampler		= &mPixelState->mSampler[pixelId];
			Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
			Ray cameraRay		= mCamera->getRay(pixelCoord, frameSize, sampler);
			cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		},
		gpContext->cudaStream);
}

void EARSPathTracer::controlExp() {
#define EXP_DELTA_TIME(x) (CpuTimer::calcDuration(x, CpuTimer::getCurrentTimePoint()) * 1e-3)
	static int sTrainState = -1;

	if (mExpState == 0) {
		Log(Info, "[Exp] Start Train for %g seconds", mExpTrainTime);
		mExpStartTime	= CpuTimer::getCurrentTimePoint();
		mAutoBuild		= true;
		mEnableLearning = true;

		sTrainState = 0;
		mExpState	= 1;

		if (mExpUseUBS) {
			mExpTrainTime -= 10; // left for Mix-Depth
		}

	} else if (mExpState == 1) {
		// train
		if (EXP_DELTA_TIME(mExpStartTime) >= mExpTrainTime) {

			if (sTrainState == 0) {
				// stop training
				mAutoBuild		= false;
				mEnableLearning = false;

				if (sGuidingTrainedFrames >= 0.5f * sTrainFramesThisIteration) {
					// update
					mOctree->refine(true);
					sGuidingTrainedFrames = 0;
				}

				if (mExpUseUBS) {
					UBSResetAndBeginSearch();
					sTrainState = 1;
				} else {
					sTrainState = 2;
				}
			} else if (sTrainState == 1) {
				// Mix-Depth ：~10s
				if (!mUBSSearch) {
					mUseBestStrategy = true;
					sTrainState		 = 2;
				}
			} else if (sTrainState == 2) {
				Log(Info, "[Exp] Start Inference for %g seconds", mExpInferenceTime);
				mExpStartTime = CpuTimer::getCurrentTimePoint();
				mExpState	  = 2;
			} else {
				Log(Warning, "[Exp] Unknown state %d", sTrainState);
			}
		}
	} else {
		if (EXP_DELTA_TIME(mExpStartTime) >= mExpInferenceTime) {
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

#undef EXP_DELTA_TIME
}

void EARSPathTracer::renderInternal(RenderContext *context) {
	mShowRRSAcc += mShowRRS ? 1 : 0;
	const auto stream = gpContext->cudaStream;

	for (int sampleId = 0; sampleId < mSpp; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(
			KRR_DEVICE_LAMBDA() {
				mCurrentRayQueue(0)->reset();
				mPathState->reset(mDebugOn);
				if (mUseNNCache) {
					mTrainBuffer->clear();
				}
			},
			stream);
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
					mScatterTidQueue->reset();
					mNonSpecularTidQueue->reset();
				},
				stream);

			// [STEP#2.1] find closest intersections, filling in mScatterRayQueue and
			// hitLightQueue
			traceClosest(depth);

			const bool queryRadianceThenBreak = mShowLi && (depth >= mShowLiDepth);

			const bool needRRS = shouldGenRRS(depth);

			if (needRRS || queryRadianceThenBreak) {
				// get inferenceQueueSize
				uint *p = (uint *) (mTempCPUBuffer + mScatterRayQueueSizeInCPUBuffer);
				*p		= 0;
				cudaMemcpyAsync(p, ((byte *) mScatterRayQueue) + mScatterRayQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToHost, stream);

				p  = (uint *) (mTempCPUBuffer + mNonSpecularQueueSizeInCPUBuffer);
				*p = 0;
				cudaMemcpyAsync(
					p, ((byte *) mNonSpecularTidQueue) + mNonSpecularTidQueue->offsetOfSize(),
					sizeof(uint), cudaMemcpyDeviceToHost, stream);
			}

			handleHit();
			handleMiss();

			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			if (queryRadianceThenBreak) {
				if (mShowLiAddRadiance) {
					queryRadiance();
				}
				break;
			}
			if (depth == mMaxDepth) {
				if (mEnableLearning) {
					// update cost
					ForAllQueued(
						mScatterRayQueue, mMaxQueueSize,
						KRR_DEVICE_LAMBDA(EARSScatterRayWorkItem & w) {
							// EARS, record cost
							mPathState->recordRadiance(w.mNodeIdx, Spectrum::Zero(), false);
							mPixelState->addStatisticAtomic(w.mPixelId, w.mDepth - 1.0f,
															Spectrum::Zero());
						},
						stream);
				}
				break;
			}

			if (needRRS) {
				generateRRSNumber(depth);
			}

			// [STEP#2.3] shadow rays & scattered rays
			if (mEnableLearning) {
				handleIntersections<true>(depth);
			} else {
				handleIntersections<false>(depth);
			}
			if (mEnableNEE) {
				traceShadow();
			}
		}

		if (mEnableLearning) {

			if (mUseNNCache) {
				updateNNCache();
			}

			if (!mUseNNCache || mUseNNCacheWithTree) {
				// update the training Octree
				updateOctree();
			}
		}
	}

	// mDenoiseOnce will be update
	bool denoiseForUITest = mDenoiseTask->shouldDenoise();
	if (mEnableLearning || denoiseForUITest) {
		// after EARS iterations
		postProcessAfterInteration(context, denoiseForUITest);

		// called every frame
		// should be called after denoised task finished
		updateImageStatistics();
	}

	// UBS
	if (mUBSSearch) {
		UBSAnalyseFrame();
	}

	const auto frameSize	  = getFrameSize();
	const uint resolutionSize = frameSize[0] * frameSize[1];
	const float sppInv		  = 1.0f / mSpp;

	// for rendereImage reweighting
	const uint sppAccForRenderededImage = mRenderedImageSpp;
	const float *diff2SumPos			= mTempGPUBuffer + mDiff2SumPosInGPUBuffer;
	const uint diff2Size				= resolutionSize - (mDiff2RemoveTopK ? mTopK : 0);

	// write results of the current frame...
	CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();
	GPUParallelFor(
		resolutionSize,
		KRR_DEVICE_LAMBDA(const int pixelId) {
			Spectrum LSpectrum = mPixelState->mL[pixelId] * sppInv;
			RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
			if (mEnableClamp) {
				L = clamp(L, 0.f, mClampMax);
			}
			if (mShowLi) {
				// pass
			} else if (mShowRRS) {
				float v = mShowRRSBuffer[pixelId] / mShowRRSAcc;

				RGB c = RGB(v, v, v);
				if (mShowScalarJetOn) {
					c = colorJetMap(v, mShowScalarJetMax);
				}
				L = c;
			} else {
				const bool reWeighting = false; // reweighting is bad
				if (mEnableLearning && mIter > 0 && reWeighting) {
					auto *pixels = mRenderedImage->data();
					// variance reweighting
					auto oldPixel = pixels[pixelId].pixel;

					float diff2Inv	 = 1.f / (*diff2SumPos / diff2Size);
					const RGB valOld = mRenderedImage->getPixel(pixelId).head<3>();
					const RGB val =
						valOld * (sppAccForRenderededImage * diff2Inv) + L * (mSpp * diff2Inv);
					pixels[pixelId].weight += (sppAccForRenderededImage + mSpp) * diff2Inv;
					pixels[pixelId].pixel += RGBA(val, 1.0f);
				} else {
					mRenderedImage->put(RGBA(L, 1.0f), pixelId);
				}
			}

			if (mExpState == 2) {
				mExpImage->put(RGBA(L, 1), pixelId);
				frameBuffer.write(mExpImage->getPixel(pixelId), pixelId);
				return;
			}

			if (mShowRenderedImage) {
				frameBuffer.write(mRenderedImage->getPixel(pixelId), pixelId);
			} else {
				frameBuffer.write(RGBA(L, 1), pixelId);
			}
		},
		stream);

	mRenderedImageSpp += mSpp;

	if (!mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer) {
		const bool shouldDenoise = mDenoiseTask->isEnabled() && mDenoiseTask->shouldDenoise();
		mDenoiseTask->updateColorBuffer(mRenderedImage);
		mDenoiseTask->denoise(true);

		// copy the denoised image to the mRenderedImage [the same as ears impl]
		if (shouldDenoise) {
			RGB *colorBuffer = mDenoiseTask->getBuffer(DenoiseTask::DenoisedColor);
			GPUParallelFor(
				resolutionSize,
				KRR_DEVICE_LAMBDA(const int pixelId) {
					auto *pixels = mRenderedImage->data();
					pixels[pixelId].pixel =
						RGBA(colorBuffer[pixelId], 1.0f) * pixels[pixelId].weight;
				},
				stream);
		}
	}

	if (!mShowRenderedImage) {
		mDenoiseTask->writeGBuffer(frameBuffer);
	}
	// const uint frameId = getFrameIndex();
	// if (mDebugOn && (frameId % 100 == 0)) {
	//	debugInfosOutput();
	// }
}

void EARSPathTracer::render(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}

	if (mUBSSearch) {
		if (!UBSNextState()) {
			UBSEndSearch();
		}
	}

	if (mExpOn) {
		controlExp();
	}

	const auto stream = gpContext->cudaStream;

	// update the mRenderedImage if the scene/camera changes
	if (mScene->getChanges()) {
		// camera changes will enter this branch
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
		mRenderedImageSpp = 0u;

		GPUCall(KRR_DEVICE_LAMBDA() { mImageStatistic->reset(); }, stream);
		mDenoiseTask->resetState();
		mDenoiseTask->setGBufferInvalid();
	}

	PROFILE("EARS Path Tracer");

	{
		const bool trainingAndShowLi = mEnableLearning && mShowLi;
		if (trainingAndShowLi) {
			mShowLi = false;
		}
		renderInternal(context);

		if (trainingAndShowLi) {
			beginFrame(context);
			mShowLi			= true;
			mEnableLearning = false;
			renderInternal(context);
			mEnableLearning = true; // restore
		}
	}
}

KRR_DEVICE_FUNCTION RGB EARSPathTracer::colorJetMap(float vIn, float vMax) {
	RGB out = RGB(1.f, 1.f, 1.f);
	float v = vIn;
	if (v < (0.25 * vMax)) {
		out[0] = 0;
		out[1] = 4 * v / vMax;
	} else if (v < (0.5 * vMax)) {
		out[0] = 0;
		out[2] = 1 + 4 * (0.25 * vMax - v) / vMax;
	} else if (v < (0.75 * vMax)) {
		out[0] = 4 * (v - 0.5 * vMax) / vMax;
		out[2] = 0;
	} else {
		v	   = min(v, vMax);
		out[1] = 1 + 4 * (0.75 * vMax - v) / vMax;
		out[2] = 0;
	}
	return out;
}

void EARSPathTracer::updateImageStatistics() {
	PROFILE("Update image statistics");
	const auto stream		  = gpContext->cudaStream;
	const auto frameSize	  = getFrameSize();
	const auto resolutionSize = frameSize[0] * frameSize[1];
	const float sppInv		  = 1.0f / mSpp;

	// [STEP#1] calculate diff2
	{
		RGB *denoisedBuffer = (RGB *) mDenoiseTask->getBuffer(DenoiseTask::DenoisedColor);
		PROFILE("Calculate diff2");

		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				// if weight is 0, set to 0.5 as orginal impl
				RGB ref		 = RGB(0.5f);
				auto *pixels = mRenderedImage->data();
				if (pixels[pixelId].weight > 0) {
					ref = mRenderedImage->getPixel(pixelId).head<3>();
				}
				Vector3f diff =
					(mPixelState->mL[pixelId] * sppInv - ref).cwiseQuotient(ref + 1e-2f);
				if (isnan(diff.mean())) {
					printf("diff is nan!, pixelId = %d, ref = %f, L = %f\n", pixelId, ref.mean(),
						   mPixelState->mL[pixelId].mean());
				}
				mTempResolutionSizeBuffer[pixelId] = diff.cwiseProduct(diff).mean();
			},
			stream);
	}

	// [STEP#2] sum of diff2 & cost
	{
		PROFILE("Sum of diff2 & cost");

		float *diff2   = mTempResolutionSizeBuffer;
		uint diff2size = resolutionSize;
		if (mDiff2RemoveTopK) {
			// we just remove the topK of diff2(keep cost unchanged), as cost won't varies a lot
			thrust::sort(thrust::device.on(stream), diff2, diff2 + resolutionSize,
						 thrust::less<float>());
			diff2size -= mTopK;
		}

		float *sum	   = mTempGPUBuffer;
		float *sumPos  = sum + mDiff2SumPosInGPUBuffer;
		float *partSum = sum + mTempPosInGPUBuffer;
		calcSum2PassAsync<false>(diff2, sumPos, partSum, diff2size, stream);

		sumPos = sum + mCostSumPosInGPUBuffer;
		float *costArray =
			((FloatPointerWarpper *) (mTempCPUBuffer + mEARSCostGPUPointerInCPUBuffer))->mData;
		calcSum2PassAsync<false>(costArray, sumPos, partSum, resolutionSize, stream);
	}

	// [STEP#3] update
	{
		PROFILE("Update");
		const float *const topkPos	  = mTempResolutionSizeBuffer;
		const float *diff2SumPos	  = mTempGPUBuffer + mDiff2SumPosInGPUBuffer;
		const float *costSumPos		  = mTempGPUBuffer + mCostSumPosInGPUBuffer;
		const float *const topkSumPos = mTempGPUBuffer + mTopKSumPosInGPUBuffer;
		const uint diff2Size		  = resolutionSize - (mDiff2RemoveTopK ? mTopK : 0);
		GPUCall(
			KRR_DEVICE_LAMBDA() {
				// initial depth = 0, so we need to add 1
				mImageStatistic->record(*diff2SumPos / diff2Size,
										(*costSumPos / resolutionSize + 1) * sppInv);
				if (mDebugOn) {
					printf("diff2: %g\n", *diff2SumPos / diff2Size);
					printf("Cost: %f, SquareError: %f\n", mImageStatistic->getCost(),
						   mImageStatistic->getSquareError());
				}
			},
			stream);
	}
}

void EARSPathTracer::postProcessAfterInteration(RenderContext *context, bool denoiseForUITest) {
	// mAutoBuild = 1
	//  => mEnableLearning = 1
	if (mEnableLearning) {
		// check iteration ends
		const bool iterationEnd = (sGuidingTrainedFrames++) >= sTrainFramesThisIteration;
		if (!iterationEnd || !mAutoBuild) {
			return;
		}
	} else {
		if (!denoiseForUITest) {
			return;
		}
	}

	if (mEnableLearning) {
		sGuidingTrainedFrames = 0;
		++mIter;
		if (mIter % mTwiceIterInterval == mTwiceIterInterval - 1) {
			sTrainFramesThisIteration *= 2;
		}

		Log(Success, "Starting iteration #%d", mIter);
		if (mAutoBuild) {

			if (!mUseNNCache || mUseNNCacheWithTree) {
				mOctree->refine(true);
			}

			GPUCall(KRR_DEVICE_LAMBDA() { mImageStatistic->reset(); }, gpContext->cudaStream);
		}
	}

	const auto stream		  = gpContext->cudaStream;
	const auto frameSize	  = getFrameSize();
	const uint resolutionSize = frameSize[0] * frameSize[1];

	// render G-Buffer
	mDenoiseTask->renderGBuffer();

	// copy color buffer
	mDenoiseTask->setShouldUpdateColorBuffer(true);
	mDenoiseTask->resetState();
}

void EARSPathTracer::debugInfosOutput() {
	const auto stream = gpContext->cudaStream;
	CUDA_SYNC_CHECK();
	GPUCall(
		KRR_DEVICE_LAMBDA() {
			uint a = mPathState->size();
			uint b = mPathState->nAlloc;
			printf("nodes: %d/%d[%f]\n", a, b, ((float) a) / b);
		},
		stream);
	CUDA_SYNC_CHECK();
}

void EARSPathTracer::beginFrame(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) return;

	PROFILE("Begin frame");

	if (mClearRenderedImageEachFrame) {
		// only for debug
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
		mRenderedImageSpp = 0u;
	}

	cudaMemcpyAsync(mCamera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
					cudaMemcpyHostToDevice, 0);
	size_t frameIndex = getFrameIndex();
	auto frameSize	  = getFrameSize();
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const int pixelId) { // reset per-pixel sample state
			Vector2i pixelCoord		 = {pixelId % frameSize[0], pixelId / frameSize[0]};
			mPixelState->mL[pixelId] = 0;
			mPixelState->mSampler[pixelId].setPixelSample(pixelCoord, frameIndex * mSpp);
			mPixelState->mSampler[pixelId].advance(256 * pixelId + mRandomOffset);
			mPixelState->mDepth[pixelId] = 0;
		},
		gpContext->cudaStream);

	mEnableLearning = (mEnableLearning || mAutoBuild);

	if (mIter == 0) {
		sTrainFramesThisIteration = mSppPerIteration;
	}
}

void EARSPathTracer::endFrame(RenderContext *context) {}

uint EARSPathTracer::loadArrayToInt(const char *arr, std::string &strConcated) {
	bool startStrS = true;
	uint ret	   = 0;

	for (int i = 0; i < 32; ++i) {
		const char c = arr[i];
		if (c == '1') {
			strConcated += (startStrS ? "" : ", ") + std::to_string(i);
			ret |= (1 << i);
			startStrS = false;
		} else if (c != '0') {
			break;
		}
	}

	return ret;
}

void EARSPathTracer::renderUI() {
	const auto resolution	  = getFrameSize();
	const auto resolutionSize = resolution[0] * resolution[1];
	const auto stream		  = gpContext->cudaStream;

	ui::Text("EARS");
	if (ui::Button("Ouput Debug Info")) {
		mOctree->debug(nullptr);
	}

	ui::Checkbox("Debug On", &mDebugOn);

	{
		auto &io = ui::GetIO();
		if (!io.WantCaptureMouse && ImGui::IsMouseClicked(0)) {
			auto frame = getFrameSize();
			auto posx  = io.MousePos.x;
			auto posy  = io.MousePos.y;
			// flip posy
			posy = frame[1] - 1 - posy;

			mDebugIntNumber = (int) (posy * frame[0] + posx);
		}
	}

	ui::SliderInt("Debug Int Number", &mDebugIntNumber, 0, resolutionSize - 1);
	ui::Checkbox("Show Li", &mShowLi);
	if (mShowLi) {
		ui::Checkbox("Add Li Radiance", &mShowLiAddRadiance);
		ui::SliderInt("Min Depth to Query Radiance Cache", &mShowLiDepth, 0, 10);
		if (mShowLiAddRadiance) {
			static const char *sShowLiTypes[] = {"Li", "LiMean", "Li^2", "Cost"};
			ui::Combo("Li Type", (int *) &mShowLiType, sShowLiTypes, std::size(sShowLiTypes));

			if (mUseNNCacheWithTree) {
				ui::Checkbox("Show NN Cache [false means Tree Cache]", &mShowLi_NNCache);
				// when use NN cache with octree, cost is trained in the octree
				if (mShowLiType == 3) {
					mShowLi_NNCache = false;
				}
			}

			// Cost is only used for NN cache
			if (!mUseNNCache && mShowLiType == 3) {
				mShowLiType = 0; // reset to Li
			}

			if (mShowLiType == 3) {
				ui::SliderFloat("Cost Weight Coefficient", &mShowLiCostWeightCoeff, 0.0f, 1.0f);
			}
		}

		if (!mUseNNCache || (mUseNNCacheWithTree && mShowLiType == 3)) {
			ui::Checkbox("Use SamplingNode", &mShowLiSamplingNode);
			static const char *sShowLiLookUpTypes[] = {"Normal", "Ignore Invalid Node",
													   "Min Depth"};
			ui::Combo("Li Look Up Type", (int *) &mShowLiLookUpType, sShowLiLookUpTypes,
					  std::size(sShowLiLookUpTypes));
			if (mShowLiLookUpType == 2) {
				ui::SliderInt("Li Look Up Min Depth For SamplingNode", &mShowLiLookUpMaxDepth, 0,
							  10);
			}
		}

		// guard
		mShowRRS = false;
	} else {
		ui::Checkbox("Show RRS", &mShowRRS);
		if (mShowRRS) {
			bool resetRRSBuffer				  = false;
			static const char *sShowRRSMode[] = {"Before Clamp", "After Clamp", "Normalized",
												 "Real Ray"};
			resetRRSBuffer |= ui::Combo("Show RRS Mode", (int *) &mShowRRSMode, sShowRRSMode,
										int(ShowRRSMode::RRSModeCount));
			{ // depth hint
				static char sShowRRSHint[32]   = "100000";
				static bool sFirstConstructStr = true;
				static std::string sShowStr{};

				if (ui::InputText("Show RRS Depth Hint", sShowRRSHint, 64) || sFirstConstructStr) {
					sFirstConstructStr = false;
					sShowStr		   = "Show RRS At ";

					uint showWhichDepth = loadArrayToInt(sShowRRSHint, sShowStr);
					mShowRRSWhichDepth	= showWhichDepth;
				}

				ui::Text("%s", sShowStr.c_str());
			}

			ui::Checkbox("Reset RRS Buffer Each Frame", &mResetRRSBufferEachFrame);
			resetRRSBuffer |= ui::Button("Reset RRS Buffer");
			if (mResetRRSBufferEachFrame || resetRRSBuffer) {
				cudaMemsetAsync(mShowRRSBuffer, 0, sizeof(float) * resolutionSize, stream);
				mShowRRSAcc = 0;
			}
			// Jet
			ui::Checkbox("Show Scalar Jet On", &mShowScalarJetOn);
			if (mShowScalarJetOn) {
				ui::SliderFloat("Show Scalar Jet UpBound[log2]", &mShowScalarJetMaxUpBound, 0.0f,
								20.0f);
				const float jetMaxExp = pow(2, mShowScalarJetMaxUpBound);
				ui::SliderFloat("Show Scalar Jet Max", &mShowScalarJetMax, 0.1f, jetMaxExp);
				mShowScalarJetMax = clamp(mShowScalarJetMax, 0.1f, jetMaxExp);
			}
			ui::Text("[RRS] Accmulate: %d", mShowRRSAcc);
		}
	}

	if (ui::TreeNodeEx("RRS Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
		ui::SliderFloat("RRS Max", &RRS_CLAMP_MAX, 1.0f, 20.0f);
		ui::SliderFloat("RRS Min", &RRS_CLAMP_MIN, 0.05f, 1.0f);
		ui::Checkbox("RRS Normalize with Ceil(rrs)", &mRRSNormalizeUseCeil);
		if (!mRRSNormalizeUseCeil) {
			ui::SliderFloat("RRS Normalize Rate", &RRS_NORMALIZE_RATE, 0.1f, 0.95f);
		}
		ui::Checkbox("Donnot Normalize When Less Than One", &mDonnotNormalizeWhenLessThanOne);
		ui::TreePop();
	}

	mOctree->renderUI();

	if (mUseNNCache) {
		if (ui::TreeNodeEx("Neural Network", ImGuiTreeNodeFlags_DefaultOpen)) {
			static const char *sLossModes[] = {"None", "All", "L", "Var", "Cost"};
			if (ui::Combo("Loss Mode", (int *) &mShowLossIndex, sLossModes,
						  std::size(sLossModes))) {
				mLoss->update_hyperparams({{"show_loss_index_ears", mShowLossIndex}});
			}

			ui::Text("Current step: %d; %d samples; loss: %f", sNumLossSamples, sNumTrainingSamples,
					 sCurLossScalar.emaVal());
			ui::Text("batches: min(%d, %d)", sNumTrainingSamples / mBatchSize + 1, mBatchPerFrame);
			ui::PlotLines("Loss graph", sLossGraph.data(), min(sNumLossSamples, sLossGraph.size()),
						  sNumLossSamples < LOSS_GRAPH_SIZE ? 0 : sNumLossSamples % LOSS_GRAPH_SIZE,
						  0, FLT_MAX, FLT_MAX, ImVec2(0, 50));

			if (ui::Checkbox("Use NN Cache with Octree", &mUseNNCacheWithTree)) {
				mLoss->update_hyperparams({{"stop_train_cost", mUseNNCacheWithTree}});
			}

			ui::TreePop();
		}
	}

	if (ui::TreeNode("Auxiliary Buffers")) {
		ui::Checkbox("Show rendered image", &mShowRenderedImage);
		ui::Checkbox("Clear Rendered Image Each Frame", &mClearRenderedImageEachFrame);
		if (ui::Button("Reset Rendered Image")) {
			cudaStreamSynchronize(gpContext->cudaStream);
			mRenderedImage->reset();
			mRenderedImageSpp = 0u;
		}
		if (!mShowRenderedImage) {
			mDenoiseTask->renderUI();
		}
		ui::TreePop();
	}

	if (ui::TreeNode("ImageStatistics")) {
		ui::Checkbox("Enable Outlier Removal", &mDiff2RemoveTopK);
		if (mDiff2RemoveTopK) {
			ui::SliderInt("Outlier Number", &mTopK, 1, 10000);
		}
		if (ui::Button("Reset ImageStatistics")) {
			GPUCall(KRR_DEVICE_LAMBDA() { mImageStatistic->reset(); }, gpContext->cudaStream);
		}
		ui::Checkbox("Use current Statistics", &mUseCurrentImageStatistic);
		ui::TreePop();
	}

	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &mSpp);
	ui::InputInt("Max bounces", &mMaxDepth, 1);
	ui::SliderFloat("Russian roulette", &mProbRR, 0, 1);
	ui::Checkbox("Enable NEE", &mEnableNEE);

	ui::Text("Octree");
	ui::Checkbox("Auto rebuild", &mAutoBuild);

	if (sTrainOneStep) {
		mAutoBuild		= false;
		mEnableLearning = false;
		sTrainOneStep	= false;
	}
	if (!mAutoBuild) {
		ui::Checkbox("Enable learning", &mEnableLearning);
		if (ui::Button("Traing One Step")) {
			mAutoBuild		= true;
			mEnableLearning = true;
			sTrainOneStep	= true;
		}
	}

	ui::Checkbox("Enable RRS", &mEnableRRS);
	ui::Checkbox("Enable RRS After PreTraining", &mEnableRRSAfterPreTraining);
	ui::SliderInt("PreTraining Iterations", &mPreTrainingIterations, 3, 10);
	ui::Checkbox("Use the accumulated buffer instead of the denoised buffer",
				 &mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer);
	if (mEnableRRSAfterPreTraining) {
		mEnableRRS = (mIter >= mPreTrainingIterations);
	}

	if (mEnableRRS) {
		if (ui::TreeNodeEx("Best RRS Strategy")) {
			// red text
			ui::TextColored(ImVec4(1, 0, 0, 1), "Now Use This when Training is OFF!");
			ui::Checkbox("Use Best Strategy", &(mUseBestStrategy));
			if (ui::Checkbox("Search for Best Stategy", &(mUBSSearch))) {
				if (mUBSSearch) {
					Log(Info, "Search For Best RRS Statagy");
					UBSResetAndBeginSearch();
				}
			}
			if (mUBSSearch) {
				ui::Text("Search State: %d/%d", mUBSSearchState, mUBSMaxState);
			} else {
				if (ui::Button("Log Best State")) {
					Log(Info, "Best RRS Stratagy: %s Efficiency: %f", mUBSArrayChar,
						mUBSBestEfficiency);
					for (int i = 0; i < mMaxDepth; ++i) {
						printf("%d", mUBSArray[i]);
					}
					printf("\n");
					for (int i = 0; i < mMaxDepth; ++i) {
						printf("%c", mUBSArrayChar[i]);
					}
					printf("\n");
				}

				// disable changing when use best strategy
				if (mUseBestStrategy) {
					ui::BeginDisabled();
				}

				if (ui::InputText("Set Best Strategy", mUBSArrayChar, 64)) {
					UBSUpdateAccordingArrayChar();
				}

				if (mUseBestStrategy) {
					ui ::EndDisabled();
				}
			}
			ui::TreePop();
		}
	}

	ui::Combo("RRS Method", (int *) &mRRSMethod, sRRSMethods, (int) RRSKind::Count);

	ui::Text("Current iteration: %d", mIter);
	ui::Text("Frames this iteration: %d / %d", sGuidingTrainedFrames, sTrainFramesThisIteration);
	ui::ProgressBar((float) sGuidingTrainedFrames / sTrainFramesThisIteration);
	if (ui::TreeNode("Advanced guiding options")) {
		if (ui::Button("Reset Octree/NN")) {
			resetOctree();
		}
		ui::DragInt("Spp per pass", &mSppPerIteration, 1, 1, 100);
		ui::TreePop();
	}
	ui::Checkbox("Clamping pixel value", &mEnableClamp);
	if (mEnableClamp) {
		ui::DragFloat("Max:", &mClampMax, 1, 500);
	}
}

void EARSPathTracer::resetOctree() {
	mOctree->initialize(mScene->getBoundingBox(), mOctreeInitDepth);

	CUDA_SYNC_CHECK();
	mRenderedImage->reset(); // TODO: should be reset?
	mRenderedImageSpp = 0u;

	mIter = sGuidingTrainedFrames = 0;
	mEnableRRS					  = false;
	CUDA_SYNC_CHECK();

	if (mUseNNCache) {
		resetNN(getFrameSize(), 0, 0);

		// return to the initial state
		std::fill(sLossGraph.begin(), sLossGraph.end(), 0);
		sNumLossSamples		= 0;
		sNumTrainingSamples = 0;
		sCurLossScalar		= Ema{Ema::Type::Time, 50};
	}
}

void EARSPathTracer::finalize() {
	Allocator &alloc = *gpContext->alloc;
	if (mDebugStats) {
		cudaFree(mDebugStats);
	}
	if (mCamera) {
		cudaFree(mCamera);
	}
	if (mTempGPUBuffer) {
		cudaFree(mTempGPUBuffer);
	}
	if (mTempCPUBuffer) {
		delete[] mTempCPUBuffer;
	}
	if (mTempResolutionSizeBuffer) {
		cudaFree(mTempResolutionSizeBuffer);
	}
	if (mRRSArray) {
		cudaFree(mRRSArray);
	}
	if (mRRSArrayCeil) {
		cudaFree(mRRSArrayCeil);
	}
	if (mOctree) {
		alloc.delete_object(mOctree);
	}
	if (mImageStatistic) {
		cudaFree(mImageStatistic);
	}
	if (mShowRRSBuffer) {
		cudaFree(mShowRRSBuffer);
	}
	if (mExpRayCounter) {
		cudaFree(mExpRayCounter);
	}

	cudaDeviceSynchronize();
	if (mTrainBuffer) {
		alloc.delete_object(mTrainBuffer);
	}

	// UBS
	if (mUBSArray) {
		delete[] mUBSArray;
	}
	if (mUBSArrayChar) {
		delete[] mUBSArrayChar;
	}

	if (mUBSErrorBufferGPU) {
		cudaFree(mUBSErrorBufferGPU);
	}
}

// NN Cache

void EARSPathTracer::resetNN(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples) {
	const cudaStream_t stream = gpContext->cudaStream;

	const int resolutionSize = frameSize[0] * frameSize[1];
	tcnn::free_gpu_memory_arena(stream);

	if (mFirstLoadConfig) {
		mLossClampOn	   = mConfig.value("clamp_on", mLossClampOn);
		mLossClampMax	   = mConfig.value("clamp_max", mLossClampMax);
		mTrainSigmaOrX2	   = mConfig.value("train_sigma", true);
		mLossShowLossIndex = mConfig.value("show_loss_index", mLossShowLossIndex);

		// assert
		if (mTrainSigmaOrX2 == true) {
			Log(Fatal, "Here, train_sigma should be false");
		}
	}

	{ // level 0
		Log(Info, "Resetting NRRS network [level0]");

		auto &config = mConfig["level0"];

		const json &encoding_config	 = config["encoding"];
		const json &optimizer_config = config["optimizer"];
		json &network_config		 = config["network"];
		const json &loss_config		 = config["loss"];

		const uint32_t n_neurons	= config["n_neurons"];
		network_config["n_neurons"] = n_neurons;

		mOptimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
		mLoss.reset(tcnn::create_loss<precision_t>(loss_config));
		mNetwork = std::make_shared<NetworkWithInputEncoding>(LL2NET_DIM_INPUT, LL2NET_DIM_OUTPUT,
															  encoding_config, network_config);
		mTrainer = std::make_shared<Trainer_LL2>(mNetwork, mOptimizer, mLoss, KRR_DEFAULT_RND_SEED);

		const auto paddedOutputWidth = mNetwork->padded_output_width();
		Log(Info, "Network has a padded output width of %d", paddedOutputWidth);

		mLoss->update_hyperparams({
			{"stop_train_cost", mUseNNCacheWithTree},
			{"clamp_on", mLossClampOn},
			{"clamp_max", mLossClampMax},
			{"train_sigma", mTrainSigmaOrX2},
			{"step", (uint32_t) 1}, // 0: only L, 1: L & L2
		});
	}

	resizeNN(frameSize, maxQueueSize, maxTrainingSamples);

	mFirstLoadConfig = false;
}

void EARSPathTracer::resizeNN(const Vector2i &frameSize, int maxQueueSize, int maxTrainingSamples) {
	const int resolutionSize = frameSize[0] * frameSize[1];
	if (!mNetIsInitialized) {
		// initialize the network
		mNetIsInitialized = true;
		resetNN(frameSize, maxQueueSize, maxTrainingSamples);
		return;
	}

	if (maxTrainingSamples) {
		mMaxTrainingSamples = maxTrainingSamples;
	}

	if (resolutionSize && maxQueueSize && maxTrainingSamples) {
		sInferenceInputBuffer  = tcnn::GPUMemory<float>(mMaxQueueSize * LL2NET_DIM_INPUT);
		sInferenceOutputBuffer = tcnn::GPUMemory<float>(mMaxQueueSize * LL2NET_DIM_OUTPUT);
	} else {
		if (maxQueueSize || maxTrainingSamples) {
			Log(Fatal, "(maxQueueSize, maxTrainingSamples) should be both 0 or all non-zero");
		}
	}
}

void EARSPathTracer::updateNNCache() {
	PROFILE("Training NN");
	const auto stream = gpContext->cudaStream;

	// [1] generate training data
	const auto frameSize	 = getFrameSize();
	const int resolutionSize = frameSize[0] * frameSize[1];

	const auto sceneAABB = mScene->getBoundingBox();

	GPUParallelFor(
		resolutionSize,
		KRR_DEVICE_LAMBDA(const int tid) {
			const uint pathStateLength = mPathState->size();

			int itemIdx = tid;

			while (itemIdx < pathStateLength) {
				EARSNetworkInput input	 = {};
				EARSNetworkOutput output = {};

				EARSRadianceRecordItem item = mPathState->operator[](itemIdx);

				RGB thpRGB	   = item.mThp.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
				const RGB LRGB = item.mL.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

				input.mPos		 = normalizeSpatialCoord(item.mPos, sceneAABB);
				input.mDir		 = utils::worldToLatLong(item.mDir); // 3d - > 2d
				input.mRoughness = warp_roughness(item.mRoughness);

				RGB L = RGB::Zero();
				for (int ch = 0; ch < RGB::dim; ch++) {
					if (thpRGB[ch] > M_EPSILON) {
						L[ch] = LRGB[ch] / thpRGB[ch];
					}
				}
				L = max(L, RGB::Zero());

				output.mL = L;
				// output.mVar	 = L * L; // in fact, not used
				output.mCost = item.mCost;

				if (!(input.mPos.hasNaN() || input.mDir.hasNaN() || L.hasNaN())) {
					mTrainBuffer->push(input, output);
				} else {
					printf("[generate_training_data] meet nan, pos = (%f, %f, %f), "
						   "dir = (%f, %f), L = (%f, %f, %f)\n",
						   input.mPos.x(), input.mPos.y(), input.mPos.z(), input.mDir.x(),
						   input.mDir.y(), L[0], L[1], L[2]);
				}
				itemIdx += resolutionSize;
			}
		},
		stream);

	// [2] train
	cudaStreamSynchronize(stream); // [TODO] if error, uncomment this line
	sNumTrainingSamples	 = mTrainBuffer->size();
	const auto inputPos	 = mTrainBuffer->inputs();
	const auto outputPos = mTrainBuffer->outputs();

	uint numTrainBatches = min((uint) sNumTrainingSamples / mBatchSize + 1, mBatchPerFrame);

	using namespace tcnn;

	float loss = 0.0f;
	for (int iter = 0; iter < numTrainBatches; iter++) {
		size_t localBatchSize = min(sNumTrainingSamples - iter * mBatchSize, (size_t) mBatchSize);
		localBatchSize -= localBatchSize % 128;

		// [TODO] in fact, we should not drop any samples
		// drop the unaligned samples
		const int localBSPad = previous_multiple<int>(localBatchSize, BATCH_SIZE_GRANULARITY);
		if (localBSPad <= 0) {
			Log(Info, "Drop the small batch: %d", localBatchSize);
			continue;
		}

		const uint dataOffset = iter * mBatchSize;
		float *inputData	  = (float *) (inputPos + dataOffset);
		float *outputData	  = (float *) (outputPos + dataOffset);

		GPUMatrix<float> networkInputs(inputData, LL2NET_DIM_INPUT, localBSPad);
		GPUMatrix<float> networkOutputs(outputData, LL2NET_DIM_OUTPUT, localBSPad);
		{
			PROFILE("Train step");
			auto ctx = mTrainer->training_step(stream, networkInputs, networkOutputs);

			if (mShowLossIndex != 0) {
				loss += mTrainer->loss(stream, *ctx);
			}
		}
	}
	// sCurLossScalar.update(loss);
	sCurLossScalar.update(loss / numTrainBatches);
	sLossGraph[sNumLossSamples++ % LOSS_GRAPH_SIZE] = sCurLossScalar.emaVal();
}

KRR_REGISTER_PASS_DEF(EARSPathTracer);
NAMESPACE_END(krr)