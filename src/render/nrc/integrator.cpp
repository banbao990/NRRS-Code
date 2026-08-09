#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "device/gpustd.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include "nrcparameters.h"
#include "integrator.h"
#include "nrctrain.h"

#include "render/profiler/profiler.h"
#include "util/ema.h"
#include "util/film.h"

NAMESPACE_BEGIN(krr)
extern "C" char NRC_BB_PTX[];
using namespace tcnn;

template <typename T> using GPUMatrix = tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor>;

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

NRCPathTracer::NRCPathTracer(Scene &scene) {
	initialize();
	setScene(std::shared_ptr<Scene>(&scene));
}

template <typename... Args>
KRR_DEVICE_FUNCTION void NRCPathTracer::debugPrint(uint pixelId, const char *fmt, Args &&...args) {
	if (pixelId == mDebugPixel) {
		printf(fmt, std::forward<Args>(args)...);
	}
}

void NRCPathTracer::initialize() {
	Allocator &alloc = *gpContext->alloc;
	mMaxQueueSize	 = getFrameSize()[0] * getFrameSize()[1];
	CUDA_SYNC_CHECK(); // necessary, preventing kernel accessing memories tobe free'ed...
	for (int i = 0; i < 2; i++) {
		if (mRayQueue[i]) {
			mRayQueue[i]->resize(mMaxQueueSize, alloc);
		} else {
			mRayQueue[i] = alloc.new_object<NRCRayQueue>(mMaxQueueSize, alloc);
		}
	}
	if (mMissRayQueue) {
		mMissRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mMissRayQueue = alloc.new_object<NRCMissRayQueue>(mMaxQueueSize, alloc);
	}
	if (mHitLightRayQueue) {
		mHitLightRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mHitLightRayQueue = alloc.new_object<NRCHitLightRayQueue>(mMaxQueueSize, alloc);
	}
	if (mShadowRayQueue) {
		mShadowRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mShadowRayQueue = alloc.new_object<SWPTShadowRayQueue>(mMaxQueueSize, alloc);
	}
	if (mScatterRayQueue) {
		mScatterRayQueue->resize(mMaxQueueSize, alloc);
	} else {
		mScatterRayQueue = alloc.new_object<NRCScatterRayQueue>(mMaxQueueSize, alloc);
	}
	if (mInferenceQueue) {
		mInferenceQueue->resize(mMaxQueueSize, alloc);
	} else {
		mInferenceQueue = alloc.new_object<NRCInferenceQueue>(mMaxQueueSize, alloc);
	}

	if (mPixelState) {
		mPixelState->resize(mMaxQueueSize, alloc);
	} else {
		mPixelState = alloc.new_object<SWPTPixelStateBuffer>(mMaxQueueSize, alloc);
	}
	if (mPathState) {
		mPathState->resize(mMaxQueueSize, alloc);
	} else {
		mPathState = alloc.new_object<NRCPathPixelStateBuffer>(mMaxQueueSize, alloc);
	}
	if (!mTrainBuffer) {
		mTrainBuffer = alloc.new_object<NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput>>(
			NRC_TRAIN_BUFFER_SIZE);
	}
	cudaDeviceSynchronize();
	if (!mCamera) {
		mCamera = alloc.new_object<Camera::CameraData>();
	}
	CUDA_SYNC_CHECK();
}

void NRCPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");

	static LaunchParameters<NRCPathTracer> params = {};

	params.mTraversable		 = mBackend->getRootTraversable();
	params.mSceneData		 = mBackend->getSceneData();
	params.mCurrentRayQueue	 = mCurrentRayQueue(depth);
	params.mMissRayQueue	 = mMissRayQueue;
	params.mHitLightRayQueue = mHitLightRayQueue;
	params.mScatterRayQueue	 = mScatterRayQueue;
	params.mNextRayQueue	 = mNextRayQueue(depth);
	mBackend->launch(params, "Closest", mMaxQueueSize, 1, 1);
}

void NRCPathTracer::traceShadow(const bool isTraining) {
	PROFILE("Trace shadow rays");
	static LaunchParameters<NRCPathTracer> params = {};

	params.mTraversable	   = mBackend->getRootTraversable();
	params.mSceneData	   = mBackend->getSceneData();
	params.mShadowRayQueue = mShadowRayQueue;
	params.mPixelState	   = mPixelState;
	params.guidedState	   = mPathState;
	params.trainState	   = mGuiding.mTrainState;
	params.isTraining	   = isTraining;
	mBackend->launch(params, "Shadow", mMaxQueueSize, 1, 1);
}

template <bool TIsTraining> void NRCPathTracer::handleEmissiveHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		mHitLightRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const NRCHitLightWorkItem &w) {
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
			if (!TIsTraining) {
				mPixelState->addRadiance(w.mPixelId, contrib);
			} else {
				mPathState->recordRadiance(w.mPixelId, contrib);
			}
		},
		gpContext->cudaStream);
}

template <bool TIsTraining> void NRCPathTracer::handleMiss() {
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
			if (!TIsTraining) {
				mPixelState->addRadiance(w.mPixelId, contrib);
			} else {
				mPathState->recordRadiance(w.mPixelId, contrib);
			}
		},
		gpContext->cudaStream);
}

template <bool TIsTraining, bool donNotTerminate>
void NRCPathTracer::handleIntersections(const int depth) {
	PROFILE("Process intersections");
	const float ternimationC = mGuiding.mStopC;

	ForAllQueued(
		mScatterRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(NRCScatterRayWorkItem & w) {
			// TODO: maybe sort will make higher coherence
			// maybe move to Closest hit shader is also ok
			if (TIsTraining && w.mDepth == mMaxDepth) {
				return;
			}
			Sampler sampler = &mPixelState->mSampler[w.mPixelId];
			// decide whether to terminate here (Russian Roulette).
			if (sampler.get1D() >= mProbRR) {
				return;
			}
			w.mThp /= mProbRR;

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
				// check should get the L_i from network or not
				const float mAn		 = w.mAn + sqrt(w.mAnEle / sample.pdf);
				const bool terminate = !sample.isDelta() && (mAn * mAn > ternimationC * w.mA0);
				Vector3f wiWorld	 = intr.toWorld(sample.wi);
				Vector3f thp		 = w.mThp * sample.f * fabs(sample.wi[2]) / sample.pdf;

				if (any(thp)) {
					// TODO: for test
					if (terminate && !donNotTerminate) {
						if (TIsTraining) {
							// TODO: now just terminate the path,
							// NRC get the radiance from the Network
						} else {
							NRCInferenceWorkItem input = {};

							input.mDir		 = wiWorld;
							input.mPos		 = offsetRayOrigin(intr.p, intr.n, wiWorld);
							input.mPixelId	 = w.mPixelId;
							input.mThp		 = thp;
							input.mA0		 = w.mA0;
							input.mDepth	 = w.mDepth + 1;
							input.mNormal	 = utils::worldToLatLong(intr.n);
							input.mRoughness = intr.sd.roughness;
							mInferenceQueue->push(input);
						}
					} else {
						NRCRayWorkItem r = {};
						Vector3f p		 = offsetRayOrigin(intr.p, intr.n, wiWorld);
						r.mBsdfType		 = sample.flags;
						r.mPdf			 = sample.pdf;
						r.mRay			 = {p, wiWorld};
						r.mCtx			 = {intr.p, intr.n};
						r.mPixelId		 = w.mPixelId;
						r.mDepth		 = w.mDepth + 1;
						r.mThp			 = thp;
						// update nrc termination heuristic
						r.mA0 = w.mA0;
						r.mAn = mAn;
						mNextRayQueue(depth)->push(r);

						if (TIsTraining) {
							Spectrum initRadiance = Spectrum::Zero();
							mPathState->incrementDepth(w.mPixelId, Ray{intr.p, wiWorld}, r.mThp,
													   initRadiance, sample.isDelta(), intr);
						}
					}
				}
			}
		},
		gpContext->cudaStream);
}

void NRCPathTracer::generateCameraRays(int sampleId) {
	PROFILE("Generate mCamera rays");
	NRCRayQueue *cameraRayQueue = mCurrentRayQueue(0);
	auto frameSize				= getFrameSize();
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

void NRCPathTracer::generateCameraRaysTraining() {
	PROFILE("Generate mCamera rays for training");
	NRCRayQueue *cameraRayQueue = mCurrentRayQueue(0);
	auto frameSize				= getFrameSize();

	// TODO: in fact, we have this queue size in the inferenceStep(), so we do not have to sync
	cudaStreamSynchronize(gpContext->cudaStream); // inferenceQueue is allocated in G/CPU
	int numInferenceSamples = mInferenceQueue->size();

	ForAllQueued(
		mInferenceQueue, numInferenceSamples,
		KRR_DEVICE_LAMBDA(const NRCInferenceWorkItem &item) {
			// TODO: generate training pixels, now all
			if (!mGuiding.isTrainingPixel(item.mPixelId)) {
				return;
			}
			NRCRayWorkItem w;
			w.mRay.origin = item.mPos;
			w.mRay.dir	  = item.mDir;
			w.mA0		  = item.mA0;
			w.mAn		  = 0;
			w.mDepth	  = item.mDepth;
			w.mThp		  = Spectrum::Ones();
			// w.mThp	  = item.mThp; // FIXME: just for test the L_i
			w.mPixelId = item.mPixelId;
			cameraRayQueue->push(w);
		},
		gpContext->cudaStream);
}

void NRCPathTracer::resize(const Vector2i &size) {
	if (size[0] * size[1] > NRC_MAX_RESOLUTION)
		Log(Fatal, "Currently maximum number of pixels is limited to %d", NRC_MAX_RESOLUTION);
	RenderPass::resize(size);
	initialize(); // need to resize the queues
}

void NRCPathTracer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	if (!mBackend) {
		mBackend	= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(NRC_BB_PTX)
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

void NRCPathTracer::beginFrame(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}
	PROFILE("Begin frame");
	auto frameSize = getFrameSize();
	cudaMemcpy(mCamera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
			   cudaMemcpyHostToDevice);
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(int pixelId) {
			// reset per-pixel radiance & sample state
			Vector2i pixelCoord		 = {pixelId % frameSize[0], pixelId / frameSize[0]};
			mPixelState->mL[pixelId] = 0;
			mPixelState->mSampler[pixelId].setPixelSample(pixelCoord, mFrameId * mSamplesPerPixel);
			mPixelState->mSampler[pixelId].advance(256 * pixelId);
			mPathState->reset(pixelId);
		},
		gpContext->cudaStream);

	mGuiding.beginFrame();
}

void NRCPathTracer::render(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}
	PROFILE("NRC Path Tracer");
	constexpr bool isTraining = false;
	const auto frameSize	  = getFrameSize();

	for (int sampleId = 0; sampleId < mSamplesPerPixel; sampleId++) {
		// [STEP#1] generate mCamera / primary rays
		GPUCall(
			KRR_DEVICE_LAMBDA() {
				mCurrentRayQueue(0)->reset();
				mTrainBuffer->clear();
				mInferenceQueue->reset();
			},
			gpContext->cudaStream);

		GPUParallelFor(
			mMaxQueueSize,
			KRR_DEVICE_LAMBDA(int pixelId) {
				Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
				mPathState->reset(pixelId);
			},
			gpContext->cudaStream);

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
			handleEmissiveHit<isTraining>();
			handleMiss<isTraining>();
			// [Sidestory] break on maximum bounce, but after handling emissive intersections.
			if (depth == mMaxDepth) {
				break;
			}
			// [STEP#2.3] handle intersections and shadow rays

			// tracing shadow rays should be before increment depth (the NEE contribution not
			// included in current vertex)
			handleIntersections<isTraining, false>(depth);
			if (mEnableNEE) {
				traceShadow(isTraining);
			}
		}

		// 1spp end

		// Network Infer
		// TODO: check if hit the light, calculate the MIS weight
		inferenceStep();

		// Network Training
		if (!mGuiding.mStopTraining) {
			trainStep();
		}
	}

	CudaRenderTarget cudaFrame = context->getColorTexture()->getCudaRenderTarget();
	if (!mDebugOn) {
		// write results of the current frame...
		GPUParallelFor(
			mMaxQueueSize,
			KRR_DEVICE_LAMBDA(int pixelId) {
				Spectrum LSpectrum = mPixelState->mL[pixelId] / float(mSamplesPerPixel);
				RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
				if (mEnableClamp) {
					L = clamp(L, 0.f, mClampMax);
				}
				cudaFrame.write(RGBA(L, 1), pixelId);
			},
			gpContext->cudaStream);
	} else {
		// write the depth
		auto inferData = sInferenceOutputBuffer.data();
		ForAllQueued(
			mInferenceQueue, NRC_MAX_RESOLUTION,
			KRR_DEVICE_LAMBDA(const NRCInferenceWorkItem &w) {
				int i = blockIdx.x * blockDim.x + threadIdx.x;
				if (i >= mInferenceQueue->size()) {
					return;
				}
				float dd = (w.mDepth == mDebugInt) ? 1.0f : 0;
				cudaFrame.write(RGBA(dd, dd, dd, 1), w.mPixelId);
			},
			gpContext->cudaStream);
	}
}

void NRCPathTracer::endFrame(RenderContext *context) { mFrameId++; }

void NRCPathTracer::finalize() {}

void NRCPathTracer::NRCParams::renderUI() {
	ui::Checkbox("Stop Training", &mStopTraining);
	mTrainState.renderUI();
	static float lr = mOptimizer->learning_rate();
	if (ui::DragFloat("Learning rate", &lr, 1e-6f, 0, 1e-1, "%.6f")) {
		mOptimizer->set_learning_rate(lr);
	}
	if (ui::CollapsingHeader("Advanced training options")) {
		if (ui::InputInt("Train pixel stride", (int *) &mTrainState.trainPixelStride, 1)) {
			mTrainState.trainPixelStride = max(1U, mTrainState.trainPixelStride);
		}
		if (ui::InputInt("Train batch size", (int *) &mBatchSize, 1)) {
			mBatchSize = max(1U, min(mBatchSize, (uint) NRC_TRAIN_BATCH_SIZE));
		}
		if (ui::InputInt("Batch per frame", (int *) &mBatchPerFrame, 1, 1)) {
			mBatchPerFrame = max(0U, mBatchPerFrame);
		}
	}
	ui::Text("Current step: %d; %d samples; loss: %f", sNumLossSamples, sNumTrainingSamples,
			 sCurLossScalar.emaVal());
	ui::PlotLines("Loss graph", sLossGraph.data(), min(sNumLossSamples, sLossGraph.size()),
				  sNumLossSamples < LOSS_GRAPH_SIZE ? 0 : sNumLossSamples % LOSS_GRAPH_SIZE, 0,
				  FLT_MAX, FLT_MAX, ImVec2(0, 50));
}

void NRCPathTracer::renderUI() {
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &mSamplesPerPixel);
	ui::InputInt("Max bounces", &mMaxDepth, 1);
	ui::SliderFloat("Russian roulette", &mProbRR, 0, 1);
	ui::Checkbox("Enable NEE", &mEnableNEE);
	if (mGuiding.mNetwork) {
		ui::Text("NRCParams");
		mGuiding.renderUI();
	}
	ui::Text("Debugging");
	ui::Checkbox("Debug On", &mDebugOn);
	ui::InputInt("DebugInt", &mDebugInt);
	ui::Checkbox("Debug output", &mDebugOutput);
	if (mDebugOutput) {
		ui::SameLine();
		ui::InputInt("Debug pixel:", (int *) &mDebugPixel);
	}
	ui::Checkbox("Train debug", &mTrainDebug);
	ui::SameLine();
	if (ui::Button("Train one step")) {
		mOneStep = 1;
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

void NRCPathTracer::resetNetwork(json config) {
	using namespace tcnn;
	tcnn::free_gpu_memory_arena(gpContext->cudaStream);

	mGuiding.mConfig	   = config;
	json &encoding_config  = config["encoding"];
	json &optimizer_config = config["optimizer"];
	json &network_config   = config["network"];
	json &loss_config	   = config["loss"];

	mGuiding.mOptimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
	mGuiding.mLoss.reset(tcnn::create_loss<precision_t>(loss_config));
	mGuiding.mNetwork = std::make_shared<NetworkWithInputEncoding<precision_t>>(
		NRC_DIM_INPUT, NRC_DIM_OUTPUT, encoding_config, network_config);

	mGuiding.mTrainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
		mGuiding.mNetwork, mGuiding.mOptimizer, mGuiding.mLoss, KRR_DEFAULT_RND_SEED);

	// TODO: i do not know why padding is needed
	Log(Info, "Network has a padded output width of %d", mGuiding.mNetwork->padded_output_width());
	// now do not need padding
	sInferenceInputBuffer  = GPUMemory<float>(NRC_DIM_INPUT * NRC_MAX_INFERENCE_NUM);
	sInferenceOutputBuffer = GPUMemory<float>(NRC_DIM_OUTPUT * NRC_MAX_INFERENCE_NUM);
	// TODO: NRC_MAX_INFERENCE_NUM: 720p, use maxResolution instead

	// mGuiding.trainer->initialize_params(); // already do in the Trainer's constructor
	mGuiding.mSampler.setSeed(KRR_DEFAULT_RND_SEED);
	CUDA_SYNC_CHECK();
}

void NRCPathTracer::resetTraining() {
	mGuiding.mTrainer->initialize_params();
	sNumLossSamples = 0;

	// reurn to the initial state
	std::fill(sLossGraph.begin(), sLossGraph.end(), 0);
	sNumLossSamples		= 0;
	sNumTrainingSamples = 0;
	sCurLossScalar		= Ema{Ema::Type::Time, 50};
}

/*
 * Do inference for all intersections that needs scattering events.
 * [1] generate inference inputs and let network predict raw outputs;
 * [2] accumulate the L_i according to the pixelID
 */
void NRCPathTracer::inferenceStep() {
	PROFILE("Inference");
	const cudaStream_t &stream							 = gpContext->cudaStream;
	std::shared_ptr<Network<float, precision_t>> network = mGuiding.mNetwork;
	if (!network) {
		logFatal("Network not initialized!");
	}

	// TODO: add a GPU call to get size, then use the size after data preparation kernel
	cudaStreamSynchronize(stream);
	int numInferenceSamples = mInferenceQueue->size();
	if (numInferenceSamples == 0) {
		return;
	}

	{
		PROFILE("Data preparation");
		LinearKernel(nrc_generate_inference_data, stream, NRC_MAX_INFERENCE_NUM, mInferenceQueue,
					 sInferenceInputBuffer.data(), mScene->getBoundingBox());
	}

	int paddedBatchSize = next_multiple(numInferenceSamples, (int) BATCH_SIZE_GRANULARITY);
	GPUMatrix<float> networkInputs(sInferenceInputBuffer.data(), NRC_DIM_INPUT, paddedBatchSize);
	GPUMatrix<float> networkOutputs(sInferenceOutputBuffer.data(), NRC_DIM_OUTPUT, paddedBatchSize);
	{
		PROFILE("Network inference");
		network->inference(stream, networkInputs, networkOutputs);
	}

	{
		PROFILE("Accumulate radiance");
		float *inferData = sInferenceOutputBuffer.data();
		ForAllQueued(
			mInferenceQueue, numInferenceSamples,
			KRR_DEVICE_LAMBDA(const NRCInferenceWorkItem &w) {
				int i	   = blockIdx.x * blockDim.x + threadIdx.x;
				Spectrum L = *((Spectrum *) (inferData + i * NRC_DIM_OUTPUT));
				L *= w.mThp;
				mPixelState->addRadiance(w.mPixelId, L);
			},
			gpContext->cudaStream);
	}
}

void NRCPathTracer::renderTrainingSuffix() {
	// the same routine as render(), however, now we should record the radiance
	if (!mScene || !mMaxQueueSize) {
		return;
	}

	PROFILE("Render Training Suffix");

	constexpr bool isTraining = true;

	// [STEP#1] generate mCamera / primary rays
	GPUCall(
		KRR_DEVICE_LAMBDA() { mCurrentRayQueue(0)->reset(); }, gpContext->cudaStream);
	generateCameraRaysTraining();
	// [STEP#2] do radiance estimation recursively
	// max depth is set to the maxDepth(add judgment when the depth grow up, per ray)
	// however we only record the radiance for the NRC_MAX_TRAIN_DEPTH
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
		handleEmissiveHit<isTraining>();
		handleMiss<isTraining>();
		// [Sidestory] break on maximum bounce, but after handling emissive intersections.
		if (depth == mMaxDepth) {
			break;
		}
		// [STEP#2.3] handle intersections and shadow rays

		// tracing shadow rays should be before increment depth (the NEE contribution not
		// included in current vertex)
		handleIntersections<isTraining, true>(depth);
		if (mEnableNEE) {
			traceShadow(isTraining);
		}
	}
}

void NRCPathTracer::trainStep() {
	PROFILE("Training");
	const cudaStream_t &stream							 = gpContext->cudaStream;
	std::shared_ptr<Network<float, precision_t>> network = mGuiding.mNetwork;
	if (!network) {
		logFatal("Network not initialized!");
	}

	renderTrainingSuffix();

	uint numTrainPixels = mMaxQueueSize / mGuiding.mTrainState.trainPixelStride;
	{
		PROFILE("Data preparation");
		LinearKernel(nrc_generate_training_data, stream, numTrainPixels,
					 mGuiding.mTrainState.trainPixelOffset, mGuiding.mTrainState.trainPixelStride,
					 mTrainBuffer, mPathState, mScene->getBoundingBox());
	}

	cudaStreamSynchronize(stream);
	sNumTrainingSamples = mTrainBuffer->size();

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
		float *inputData = (float *) (mTrainBuffer->inputs() + iter * mGuiding.mBatchSize);
		NRCNetworkOutput *outputData = mTrainBuffer->outputs() + iter * mGuiding.mBatchSize;

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

KRR_REGISTER_PASS_DEF(NRCPathTracer);
NAMESPACE_END(krr)