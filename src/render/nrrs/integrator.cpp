#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/async/reduce.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <fstream>

#include "render/common/commoncudautilshost.h"

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
extern "C" char NRRS_BB_PTX[];
using namespace tcnn;

namespace {
// lossgraph and training info logging / plotting
constexpr size_t LOSS_GRAPH_SIZE = 256;
std::vector<float> sLossGraph(LOSS_GRAPH_SIZE, 0);
size_t sNumLossSamples{0};
size_t sNumTrainingSamples{0};
Ema sCurLossScalar{Ema::Type::Time, 50};
bool sShowError{false};
} // namespace

template <typename... Args>
KRR_DEVICE_FUNCTION void NRRSPathTracer::debugPrint(uint pixelId, const char *fmt, Args &&...args) {
	if (pixelId == mDebugPixel) printf(fmt, std::forward<Args>(args)...);
}

void NRRSPathTracer::initialize(bool keepRenderedImage) {
	Log(Info, "NRRSPathTracer::initialize()");
	if (!mUBSArray) {
		mUBSArray = new int[UBS_MAX_DEPTH];
		memset(mUBSArray, 0, sizeof(int) * UBS_MAX_DEPTH);
	}
	if (!mUBSArrayChar) {
		mUBSArrayChar = new char[UBS_MAX_DEPTH];
		memset(mUBSArrayChar, 0, sizeof(char) * UBS_MAX_DEPTH);
	}

	Allocator &alloc		  = *gpContext->alloc;
	cudaStream_t stream		  = gpContext->cudaStream;
	const auto frameSize	  = getFrameSize();
	const uint resolutionSize = frameSize[0] * frameSize[1];
	mMaxQueueSize			  = resolutionSize;
	// mMaxQueueSize = next_multiple((int) (1.5f * resolutionSize), (int) BATCH_SIZE_GRANULARITY);

	CUDA_SYNC_CHECK(); // necessary, preventing kernel accessing memories tobe free'ed...

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

	{ // simple or normal render mode
		if (mNRRSParams.mSimpleRenderOn) {
			for (int i = 0; i < 2; i++) {
				if (mRayQueueSimple[i]) {
					mRayQueueSimple[i]->resize(mMaxQueueSize, alloc);
				} else {
					mRayQueueSimple[i] = alloc.new_object<SWPTRayQueue>(mMaxQueueSize, alloc);
				}
			}
			if (mMissRayQueueSimple) {
				mMissRayQueueSimple->resize(mMaxQueueSize, alloc);
			} else {
				mMissRayQueueSimple = alloc.new_object<SWPTMissRayQueue>(mMaxQueueSize, alloc);
			}
			if (mHitLightRayQueueSimple) {
				mHitLightRayQueueSimple->resize(mMaxQueueSize, alloc);
			} else {
				mHitLightRayQueueSimple =
					alloc.new_object<SWPTHitLightRayQueue>(mMaxQueueSize, alloc);
			}
			if (mShadowRayQueueSimple) {
				mShadowRayQueueSimple->resize(mMaxQueueSize, alloc);
			} else {
				mShadowRayQueueSimple = alloc.new_object<SWPTShadowRayQueue>(mMaxQueueSize, alloc);
			}
			if (mScatterRayQueueSimple) {
				mScatterRayQueueSimple->resize(mMaxQueueSize, alloc);
			} else {
				mScatterRayQueueSimple =
					alloc.new_object<SWPTScatterRayQueue>(mMaxQueueSize, alloc);
			}
			mScatterRayQueueSimplePixelIdPtr = mScatterRayQueueSimple->mPixelId;
		} else {
			for (int i = 0; i < 2; i++) {
				if (mRayQueue[i]) {
					mRayQueue[i]->resize(mMaxQueueSize, alloc);
				} else {
					mRayQueue[i] = alloc.new_object<NRRSRayQueue>(mMaxQueueSize, alloc);
				}
			}
			if (mMissRayQueue) {
				mMissRayQueue->resize(mMaxQueueSize, alloc);
			} else {
				mMissRayQueue = alloc.new_object<NRRSMissRayQueue>(mMaxQueueSize, alloc);
			}
			if (mHitLightRayQueue) {
				mHitLightRayQueue->resize(mMaxQueueSize, alloc);
			} else {
				mHitLightRayQueue = alloc.new_object<NRRSHitLightRayQueue>(mMaxQueueSize, alloc);
			}
			if (mShadowRayQueue) {
				mShadowRayQueue->resize(mMaxQueueSize, alloc);
			} else {
				mShadowRayQueue = alloc.new_object<NRRSShadowRayQueue>(mMaxQueueSize, alloc);
			}
			if (mScatterRayQueue) {
				mScatterRayQueue->resize(mMaxQueueSize, alloc);
			} else {
				mScatterRayQueue = alloc.new_object<NRRSScatterRayQueue>(mMaxQueueSize, alloc);
			}
			mScatterRayQueuePixelIdPtr = mScatterRayQueue->mPixelId;
		}
	}

	if (mInferenceQueue) {
		mInferenceQueue->resize(mMaxQueueSize, alloc);
	} else {
		mInferenceQueue = alloc.new_object<NRRSInferenceQueue>(mMaxQueueSize, alloc);
	}
	if (mScatterTidQueue) {
		mScatterTidQueue->resize(mMaxQueueSize, alloc);
	} else {
		mScatterTidQueue = alloc.new_object<TidQueue>(mMaxQueueSize, alloc);
	}

	if (mPixelState) {
		mPixelState->resize(mMaxQueueSize, alloc);
	} else {
		mPixelState = alloc.new_object<NRRSPixelStateBuffer>(mMaxQueueSize, alloc);
	}

	const int maxTrainingSamples = resolutionSize * mMaxRateForPathNodesBuffer;
	if (mPathState) {
		mPathState->resize(maxTrainingSamples, alloc);
	} else {
		mPathState = alloc.new_object<NRRSPathNodesBuffer>(maxTrainingSamples, alloc);
	}
	if (mPathStateNodeIdxAtomicBuffer) {
		cudaFree(mPathStateNodeIdxAtomicBuffer);
	}
	cudaMalloc(&mPathStateNodeIdxAtomicBuffer, sizeof(uint) * maxTrainingSamples);

	if (mTrainBuffer) {
		mTrainBuffer->resize(maxTrainingSamples);
	} else {
		mTrainBuffer =
			alloc.new_object<NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0>>(
				maxTrainingSamples);
	}

	Vector2i frameSizeScaled = frameSize / mErrorImageScale;
	Log(Info, "Erorr/RenderedImage Frame size scaled: %d x %d", frameSizeScaled[0],
		frameSizeScaled[1]);

	{
		// as the renderer design, pass->initialize() only called once out side itself
		static bool mDenoiseTaskInitialized = false;
		if (!mDenoiseTaskInitialized) {
			mDenoiseTask->initialize(frameSizeScaled[0], frameSizeScaled[1]);
			mDenoiseTaskInitialized = true;
		}
	}
	int frameSizeScaledLength = frameSizeScaled[0] * frameSizeScaled[1];
	{
		if (mWeightedLBuffer) {
			cudaFree(mWeightedLBuffer);
		}
		cudaMalloc(&mWeightedLBuffer, sizeof(RGB) * frameSizeScaledLength);
		if (mWeightedLBufferCurrent) {
			cudaFree(mWeightedLBufferCurrent);
		}
		cudaMalloc(&mWeightedLBufferCurrent, sizeof(RGB) * frameSizeScaledLength);
		if (mWeightedLBufferCurrentAcc) {
			cudaFree(mWeightedLBufferCurrentAcc);
		}
		cudaMalloc(&mWeightedLBufferCurrentAcc, sizeof(uint) * frameSizeScaledLength);
		if (mNumberSamplesThisPatch) {
			cudaFree(mNumberSamplesThisPatch);
		}
		cudaMalloc(&mNumberSamplesThisPatch, sizeof(float) * frameSizeScaledLength);
	}

	// if (mErrorFactorXX) {
	//	cudaFree(mErrorFactorXX);
	// }
	// cudaMalloc(&mErrorFactorXX, sizeof(float) * frameSizeScaledLength);
	// if (mErrorFactorXY) {
	//	cudaFree(mErrorFactorXY);
	// }
	// cudaMalloc(&mErrorFactorXY, sizeof(float) * frameSizeScaledLength);

	if (!mRenderedImage) {
		mRenderedImage = alloc.new_object<MyFilm>(frameSize, mErrorImageScale, frameSizeScaled);
	} else {
		if (mRenderedImage->size() != frameSizeScaled || !keepRenderedImage) {
			mRenderedImage->resize(frameSizeScaled); // will call reset()
		}
	}

	if (!mRefImageDebug) {
		mRefImageDebug = alloc.new_object<Film>(frameSize);
		{ // [DEBUG] load
			std::string path;
			if (gpContext->getGlobalConfig().contains("reference")) {
				path = gpContext->getGlobalConfig().at("reference");
			}
			auto img	 = std::make_shared<Image>();
			bool success = img->loadImage(path, true, false);
			if (success) {
				// TODO: find out why saving an exr image yields this permutation on pixel format?
				// This should be deleted once new reference images are updated.
				auto permute = [](auto pixel) {
					Array4i p = {3, 0, 1, 2};
					auto res  = pixel;
					for (int c = 0; c < 4; c++) res[c] = pixel[p[c]];
					return res;
				};
				img->process(permute);
				RGBA *data;
				cudaMalloc(&data, sizeof(RGBA) * img->getSizeInBytes() / sizeof(RGBA));
				cudaMemcpy(data, reinterpret_cast<RGBA *>(img->data()),
						   sizeof(RGBA) * img->getSizeInBytes() / sizeof(RGBA),
						   cudaMemcpyHostToDevice);
				GPUParallelFor(
					resolutionSize,
					KRR_DEVICE_LAMBDA(const int pixelId) {
						mRefImageDebug->put(data[pixelId], pixelId);
					},
					gpContext->cudaStream);
				cudaFree(data);
				Log(Info, "NRRSPathTracer::Loaded reference image from %s.", path.c_str());
			} else {
				Log(Error, "NRRSPathTracer::Failed to load reference image from %s", path.c_str());
			}
		}
	}

	if (mNRRSParams.mRRSArray) {
		cudaFree(mNRRSParams.mRRSArray);
	}
	cudaMalloc(&mNRRSParams.mRRSArray, sizeof(float) * mMaxQueueSize);
	if (mNRRSParams.mRRSArrayCeil) {
		cudaFree(mNRRSParams.mRRSArrayCeil);
	}
	cudaMalloc(&mNRRSParams.mRRSArrayCeil, sizeof(float) * mMaxQueueSize);

	if (mUBSErrorBufferGPU) {
		cudaFree(mUBSErrorBufferGPU);
	}
	cudaMalloc(&mUBSErrorBufferGPU, sizeof(float) * resolutionSize);

	if (mRenderedImageDenoised) {
		cudaFree(mRenderedImageDenoised);
	}
	cudaMalloc(&mRenderedImageDenoised, sizeof(RGB) * resolutionSize);

	// if (mCostBuffer) {
	// cudaFree(mCostBuffer);
	//}
	// cudaMalloc(&mCostBuffer, sizeof(float) * resolutionSize);

	mShowBuffer.allocateBuffer(resolutionSize);

	if (!mCamera) {
		cudaMalloc(&mCamera, sizeof(Camera::CameraData));
	}
	if (!mNRRSParams.mTempGPUBuffer) {
		cudaMalloc(&mNRRSParams.mTempGPUBuffer, sizeof(float) * mNRRSParams.mTempGPUBufferSize);
	}
	if (!mNRRSParams.mTempCPUBuffer) {
		mNRRSParams.mTempCPUBuffer = new float[mNRRSParams.mTempCPUBufferSize];
	}

	// if (mRRSScaler) {
	//	cudaFree(mRRSScaler);
	// }
	// cudaMalloc(&mRRSScaler, sizeof(float) * RS_MAX_DEPTH);
	// GPUCall(
	//	KRR_DEVICE_LAMBDA() {
	//		for (int i = 0; i < RS_MAX_DEPTH; i++) {
	//			mRRSScaler[i] = 1.0f;
	//		}
	//		// test
	//		float multiplier = 0.9f;
	//		for (int i = 0; i < 6; ++i) {
	//			mRRSScaler[i] = multiplier;
	//			multiplier *= 0.9f;
	//		}
	//	},
	//	stream);

	cudaDeviceSynchronize();
}

void NRRSPathTracer::traceClosest(const int depth) {
	PROFILE("Trace intersect rays");
	static LaunchParameters<NRRSPathTracer> params = {};

	params.mTraversable		 = mBackend->getRootTraversable();
	params.mSceneData		 = mBackend->getSceneData();
	params.mCurrentRayQueue	 = currentRayQueue(depth);
	params.mMissRayQueue	 = mMissRayQueue;
	params.mHitLightRayQueue = mHitLightRayQueue;
	params.mScatterRayQueue	 = mScatterRayQueue;
	params.mNextRayQueue	 = nextRayQueue(depth);
	params.mInferenceQueue	 = mInferenceQueue;
	params.mScatterTidQueue	 = mScatterTidQueue;
	params.mUseRRS			 = mEnableRRS;
	params.mShowLi			 = mShowBuffer.showLi() && depth >= mShowBuffer.mShowLiDepth;
	// params.mRenderedImage	 = mRenderedImage;
	params.mSpecularBuffer = mShowBuffer.showSpecular() ? mShowBuffer.mShowBuffer : nullptr;

	mBackend->launch(params, "Closest", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mCurrentRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += rayQueue->size(); }, gpContext->cudaStream);
	}
}

void NRRSPathTracer::traceShadow(const bool isTraining) {
	PROFILE("Trace shadow rays");
	static LaunchParameters<NRRSPathTracer> params = {};

	params.mTraversable	   = mBackend->getRootTraversable();
	params.mSceneData	   = mBackend->getSceneData();
	params.mShadowRayQueue = mShadowRayQueue;
	params.mPixelState	   = mPixelState;
	params.mPathState	   = mPathState;
	params.mIsTraining	   = isTraining;
	mBackend->launch(params, "Shadow", mMaxQueueSize, 1, 1);

	if (mExpOn && mExpState == 2) {
		// record traced rays
		auto rayQueue = params.mShadowRayQueue;
		GPUCall(
			KRR_DEVICE_LAMBDA() { *mExpRayCounter += rayQueue->size(); }, gpContext->cudaStream);
	}
}

uint NRRSPathTracer::loadArrayToInt(const char *arr, std::string &strConcated) {
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

KRR_DEVICE_FUNCTION RGB NRRSPathTracer::colorJetMap(float vIn, float vMax) {
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

template <bool tIsSimpleMode>
void NRRSPathTracer::writeResultToRenderTarget(RenderContext *context) {
	CudaRenderTarget cudaFrame = context->getColorTexture()->getCudaRenderTarget();
	const auto frameSize	   = getFrameSize();
	const auto resolutionSize  = frameSize[0] * frameSize[1];
	const uint frameSizeWidth  = uint(frameSize[0]);
	const auto stream		   = gpContext->cudaStream;

	if (tIsSimpleMode) {
		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				// read rendered L
				Spectrum LSpectrum = mPixelState->mL[pixelId] / mSamplesPerPixel;
				RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
				if (mEnableClamp) {
					L = clamp(L, 0.f, mClampMax);
				}

				mRenderedImage->put(RGBA(L, 1.0f), pixelId);

				if (mExpState == 2) {
					mExpImage->put(RGBA(L, 1), pixelId);
					L = mExpImage->getPixel(pixelId).head<3>();
				}
				cudaFrame.write(RGBA(L, 1), pixelId);
			},
			stream);
	} else {
		const auto showLiType = mShowBuffer.mShowType;

		const bool isRRS	   = (showLiType == ShowDebugBuffer::ShowRRS);
		const float *rrsBuffer = isRRS ? mShowBuffer.mShowRRSBuffer : mShowBuffer.mShowBuffer;
		const uint rrsAcc	   = isRRS ? mShowBuffer.mShowRRSAcc : mShowBuffer.mShowSpecularAcc;

		const bool jetOn		   = mShowBuffer.mShowScalarJetOn;
		const float jetMax		   = mShowBuffer.mShowScalarJetMax;
		const float *errorPerPixel = mNRRSParams.mNet->getErrorPerPixel();

		const auto errorPerPixelParams		= mShowBuffer.mShowErrorPerPixelParams;
		const float *errorPerPixelSumGPUPtr = mNRRSParams.mNet->getLossSumErrorGPUPtr();

		auto &net = mNRRSParams.mNet;

		uint *netPixelDebug	  = nullptr;
		uint netPixelDebugCnt = 0;
		net->getPixelDebugBuffer(netPixelDebug, netPixelDebugCnt);

		const bool errorMultiplySamples = mNRRSParams.mNet->getPixelErrorMultiplySamples();

		Vector2i drawCenter{mDebugPixel % frameSize[0], mDebugPixel / frameSize[0]};

		const bool networkDebugTypeIsFloat = mShowBuffer.mShowNetDebugTypeIsFloat;

		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				// read rendered L
				Spectrum LSpectrum = mPixelState->mL[pixelId] / mSamplesPerPixel;
				RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
				if (mEnableClamp) {
					L = clamp(L, 0.f, mClampMax);
				}

				// save this frame
				const bool showLi = showLiType == ShowDebugBuffer::ShowLi;

				if (!mLowPowerMode) {
					if (!showLi) {
						mRenderedImage->put(RGBA(L, 1), pixelId);
					}
				}

				// show buffer
				RGB finalColor = L;
				switch (showLiType) {
					case ShowDebugBuffer::ShowWeightedL: {
						int pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);
						finalColor = mWeightedLBuffer[pixelIdScaled];
					} break;
					case ShowDebugBuffer::ShowRenderedImage: {
						finalColor = mRenderedImage->getPixel(pixelId).head<3>();
					} break;
					case ShowDebugBuffer::ShowRenderedImageDenoised: {
						int pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);
						finalColor = mRenderedImageDenoised[pixelIdScaled];
					} break;
					case ShowDebugBuffer::ShowRefImage: {
						finalColor = mRefImageDebug->getPixel(pixelId).head<3>();
					} break;
					case ShowDebugBuffer::ShowRRS:
						// fallthrough
					case ShowDebugBuffer::ShowSpecular: {
						float v = rrsBuffer[pixelId] / rrsAcc;
						RGB c	= RGB(v, v, v);
						if (jetOn) {
							c = colorJetMap(v, jetMax);
						}
						finalColor = c;
					} break;
					case ShowDebugBuffer::ShowLError: {
						// RGB LRef	  = mRefImageDebug->getPixel(pixelId).head<3>();
						RGB LRef   = mRenderedImage->getPixel(pixelId).head<3>();
						float v	   = ((L - LRef) / (LRef + RGB(1e-2))).square().mean();
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					case ShowDebugBuffer::ShowWeightedLError: {
						// RGB LRef	  = mRefImageDebug->getPixel(pixelId).head<3>();
						RGB LRef = mRenderedImage->getPixel(pixelId).head<3>();

						int pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);
						RGB weightedL = mWeightedLBuffer[pixelIdScaled];
						float v		  = ((weightedL - LRef) / (LRef + RGB(1e-2))).square().mean();
						if (errorMultiplySamples) {
							// here donnot multiply by 0.1 [scale]
							v = v / mNumberSamplesThisPatch[pixelIdScaled];
						}
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					case ShowDebugBuffer::ShowErrorPerPixel: {
						int pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);
						float v = errorPerPixel[pixelIdScaled];
						if (errorPerPixelParams.mLogScale) {
							v = log10f(v + 1);
						}
						if (errorPerPixelParams.mLargeThan) {
							v = (v > errorPerPixelParams.mLargeBound) ? 1.0f : 0;
						} else if (errorPerPixelParams.mShowRelativeToMean) {
							float mean = *errorPerPixelSumGPUPtr / resolutionSize;
							// clamp => [-20,20] => [0,1]
							v = clamp((v - mean) / (errorPerPixelParams.mShowRelativeToMeanScale),
									  -1.0f, 1.0f);
							v = (v + 1.0f) * 0.5f;
						}
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					case ShowDebugBuffer::ShowTrainingSamplesPerPixel: {
						float v	   = mPixelState->mTrainingSamples[pixelId];
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					case ShowDebugBuffer::ShowErrorPerSample: {
						int pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);

						float v = errorPerPixel[pixelIdScaled];
						v /= mPixelState->mTrainingSamples[pixelId];
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					case ShowDebugBuffer::ShowNumberSamplesPerPixel: {
						int pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);
						float v	   = float(mNumberSamplesThisPatch[pixelIdScaled]);
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					case ShowDebugBuffer::ShowNetworkDebug: {
						float v;
						if (networkDebugTypeIsFloat) {
							v = ((float *) netPixelDebug)[pixelId] / (float) netPixelDebugCnt;
						} else {
							v = ((uint *) netPixelDebug)[pixelId] / (float) netPixelDebugCnt;
						}
						finalColor = RGB(v, v, v);
						if (jetOn) {
							finalColor = colorJetMap(v, jetMax);
						}
					} break;
					default: {
						// default case, show current frame
					} break;
				}

#if 0 // DBEUG
			if (mDrawRect) {
				int x = pixelId % frameSize[0];
				int y = pixelId / frameSize[0];
				if (abs(x - drawCenter[0]) <= mDrawRectSize &&
					abs(y - drawCenter[1]) <= mDrawRectSize) {
					finalColor = RGB(1.0f, 0, 0);
				}
			}
#endif
				cudaFrame.write(RGBA(finalColor, 1), pixelId);
			},
			stream);

		mDenoiseTask->writeGBuffer(cudaFrame, mErrorImageScale);
	}
}

void NRRSPathTracer::queryNetwork() {
	PROFILE("Query network");

	const auto stream = gpContext->cudaStream;

	int inferenceQueueSize = 0;
	inferenceQueueSize =
		*(int *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mInferQueueSizeInCPUBuffer);
	if (!inferenceQueueSize) {
		Log(Info, "[queryNetwork] Sync");
		// sync as this is allocated by cudaMallocManaged()
		cudaStreamSynchronize(stream);
		inferenceQueueSize = mInferenceQueue->size();
	}

	std::shared_ptr<NRRSNetwork> &network = mNRRSParams.mNet;

	precision_t *outputPtr_L  = network->getInferenceOutputBufferPtr_L();
	precision_t *outputPtr_L2 = network->getInferenceOutputBufferPtr_L2();

	// prepare data
	network->prepareInferenceData(inferenceQueueSize, mInferenceQueue, mScene->getBoundingBox(),
								  false, nullptr, nullptr);
	// inference
	// only LL2Net, get L/L2
	network->inference(inferenceQueueSize, true, mInferenceQueue, nullptr, nullptr);

	// add mRadiance
	const int showLiType		= mShowBuffer.mShowLiType;
	const bool showLiGray		= mShowBuffer.mShowLiGray;
	const bool netOutputIsSigma = network->getTrainSigmaOrX2();
	GPUParallelFor(
		inferenceQueueSize,
		KRR_DEVICE_LAMBDA(const int tid) {
			const uint sid	   = mInferenceQueue->mScatterQueueIndex[tid];
			const uint pixelId = mScatterRayQueue->mPixelId[sid];
			const Spectrum thp = mScatterRayQueue->mThp[sid];

			const precision_t *ptr_L  = outputPtr_L + (tid * NRRS_LL2NET_DIM_OUTPUT_PADDED);
			const precision_t *ptr_L2 = outputPtr_L2 + (tid * NRRS_LL2NET_DIM_OUTPUT_PADDED);
			RGB LMean{ptr_L[0], ptr_L[1], ptr_L[2]};
			const RGB LOut2{ptr_L2[0], ptr_L2[1], ptr_L2[2]};

			const RGB LMean2 = LMean * LMean;
			RGB LSigma		 = RGB(0.0f);
			RGB L2Mean		 = RGB(0.0f);

			if (netOutputIsSigma) {
				LSigma[0] = activationSigma(LOut2[0]);
				LSigma[1] = activationSigma(LOut2[1]);
				LSigma[2] = activationSigma(LOut2[2]);
				L2Mean	  = (LSigma * LSigma + LMean2).sqrt();
			} else {
				// output is Variance
				LSigma = max(LOut2, RGB(0)).sqrt();
				L2Mean = max(RGB(0), (LOut2 + LMean2)).sqrt();

				// L2Mean = LOut2.sqrt();
				// LSigma = max(L2Mean - LMean2, RGB(0.0f)).sqrt();
			}

			RGB LArray[3] = {LMean, L2Mean, LSigma};
			Spectrum L = Spectrum::fromRGB(LArray[showLiType], {}, {}, *KRR_DEFAULT_COLORSPACE_GPU);

			L *= thp;
			if (showLiGray) {
				L = Spectrum(L.mean());
			}
			mPixelState->addRadianceAtomic(pixelId, L);
		},
		stream);
}

template <bool tIsTraining> void NRRSPathTracer::handleEmissiveHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		mHitLightRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const NRRSHitLightWorkItem &w) {
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
				mPathState->recordRadiance(w.mNodeIdx, contrib);
			}
			mPixelState->addRadianceAtomic(w.mPixelId, contrib);
		},
		gpContext->cudaStream);
}

template <bool tIsTraining> void NRRSPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	const rt::SceneData &mSceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(
		mMissRayQueue, mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const NRRSMissRayWorkItem &w) {
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
				mPathState->recordRadiance(w.mNodeIdx, contrib);
			}
			mPixelState->addRadianceAtomic(w.mPixelId, contrib);
		},
		gpContext->cudaStream);
}

template <bool tIsTraining> void NRRSPathTracer::handleIntersections(const int depth) {
	PROFILE("Process intersections");
	const auto stream	   = gpContext->cudaStream;
	const bool shouldInfer = isInferMode(depth);

	const uint frameSizeWidth = uint(getFrameSize()[0]);

	if (!shouldInfer) {
		ForAllQueued(
			mScatterRayQueue, mMaxQueueSize,
			KRR_DEVICE_LAMBDA(NRRSScatterRayWorkItem & w) {
				// TODO: maybe sort will make higher coherence
				// maybe move to Closest hit shader is also ok
				const uint tid = blockIdx.x * blockDim.x + threadIdx.x;

				Sampler sampler = &mPixelState->mSampler[tid];

				// float rrTraditional = clamp(w.mThp.mean(), 0.05f, 1.0f);
				float rrTraditional = mFixedProbRR;

				// decide whether to terminate here (Russian Roulette).
				const float rr = tIsTraining ? 1.0f : rrTraditional;
				if (sampler.get1D() >= rr) {
					return;
				}

				w.mThp /= rr;
				if (tIsTraining) {
					w.mRRS = rr;
				}

				generateScatteredRays<tIsTraining>(w, sampler, depth, frameSizeWidth);
			},
			stream);
	} else {
		GPUParallelFor(
			mMaxQueueSize,
			KRR_DEVICE_LAMBDA(const int tid) {
				if (tid >= mScatterTidQueue->size()) {
					return;
				}

				if (tid == 0) {
					// [TODO] infact, need re-start generating rays [=> GenerateRRSNumber()]
					if (mScatterTidQueue->size() > mMaxQueueSize) {
						printf("depth = %d, scatterTidQueue->size() > maxQueueSize, %d > %d\n",
							   depth, mScatterTidQueue->size(), mMaxQueueSize);
					}
				}
				const uint sid = mScatterTidQueue->mTid[tid];

				NRRSScatterRayWorkItem sitem = (*mScatterRayQueue)[sid]; // copy
				Sampler sampler				 = &mPixelState->mSampler[tid];

				sitem.mThp /= sitem.mRRS;
				generateScatteredRays<tIsTraining>(sitem, sampler, depth, frameSizeWidth);
			},
			stream);
	}
}

template <bool tIsTraining>
KRR_DEVICE_FUNCTION void NRRSPathTracer::generateScatteredRays(const NRRSScatterRayWorkItem &w,
															   Sampler &sampler, const int depth,
															   const uint frameSizeWidth) {

	if (tIsTraining) {
		// if not needed, delete following 2 lines
		int pixelIdScaled = offsetFrame2Scaled(w.mPixelId, frameSizeWidth, mErrorImageScale);
		float *numAcc	  = mNumberSamplesThisPatch + pixelIdScaled;
		atomicAdd(numAcc, 1.0f);
	}

	const SurfaceInteraction &intr = w.mIntr;
	Vector3f woLocal			   = intr.toLocal(intr.wo);
	BSDFType bsdfType			   = intr.getBsdfType();

	const bool alreadyUpdateNodeIdx = !mCopyTheRRSNode && isInferMode(depth);

	int nodeIdx = w.mNodeIdx;
	// get a node for current point, skip if specular
	if (!alreadyUpdateNodeIdx && tIsTraining && (bsdfType & BSDF_SMOOTH)) {
		// TODO: needs optimization, change sd.mWo from 3D to 2D
		// sd.mWo : view direction, normalized, last point -> current point
		// sd.pos : not normalized
		// mThp   : not multiplied by BSDF

		nodeIdx = mPathState->recordNode(w.mNodeIdx, intr.p, intr.wo, intr.sd.roughness,
										 w.mThp, // rrs performed, as copy node
										 w.mPixelId, w.mRRS, depth);
		if (nodeIdx == -1) {
			nodeIdx = w.mNodeIdx;
		}
	}

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
			NRRSShadowRayWorkItem sw = {};

			sw.mRay		= shadowRay;
			sw.mLi		= ls.L;
			sw.mPixelId = w.mPixelId;
			sw.mMaxT	= 1;
			sw.mNodeIdx = nodeIdx;
			sw.mDepth	= w.mDepth + 1;
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
		Spectrum thp	 = w.mThp * sample.f * fabs(sample.wi[2]) / sample.pdf;

		if (any(thp)) {
			NRRSRayWorkItem r = {};

			Vector3f p	= offsetRayOrigin(intr.p, intr.n, wiWorld);
			r.mBsdfType = sample.flags;
			r.mPdf		= sample.pdf;
			r.mRay		= {p, wiWorld};
			r.mCtx		= {intr.p, intr.n};
			r.mPixelId	= w.mPixelId;
			r.mDepth	= w.mDepth + 1;
			r.mNodeIdx	= nodeIdx;
			r.mThp		= thp;
			nextRayQueue(depth)->push(r);
		}
	}
}

void NRRSPathTracer::generateCameraRays(int sampleId) {
	PROFILE("Generate camera rays");
	NRRSRayQueue *cameraRayQueue = currentRayQueue(0);
	auto frameSize				 = getFrameSize();
	const uint resolutionSize	 = frameSize[0] * frameSize[1];
	const auto stream			 = gpContext->cudaStream;

	if (mAdaptiveSamplingOn) {
		// fisrt tASPixels => AS on the debug pixel region
		int cx, cy;
		if (mDebugPixel == -1) {
			cx = frameSize[0] / 2;
			cy = frameSize[1] / 2;
		} else {
			cx = mDebugPixel % frameSize[0];
			cy = mDebugPixel / frameSize[0];
		}
		int bx, by, ex, ey;
		bx		 = max(0, cx - mAdaptiveSamplingRadius);
		by		 = max(0, cy - mAdaptiveSamplingRadius);
		ex		 = min(frameSize[0], cx + mAdaptiveSamplingRadius + 1);
		ey		 = min(frameSize[1], cy + mAdaptiveSamplingRadius + 1);
		int size = (ex - bx) * (ey - by);

		int tASPixels = size * (mAdaptiveSamplingSpp - 1);
		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				int tid			= pixelId;
				Sampler sampler = &mPixelState->mSampler[tid];
				if (tid < tASPixels) {
					int idx = tid % size;
					int x	= idx % (ex - bx) + bx;
					int y	= idx / (ex - bx) + by;

					tid = y * frameSize[0] + x;
				}
				Vector2i pixelCoord = {tid % frameSize[0], tid / frameSize[0]};
				Ray cameraRay		= mCamera->getRay(pixelCoord, frameSize, sampler);
				int rayIdx			= cameraRayQueue->pushCameraRay(cameraRay, tid);

				// if in the AS region, update the thp
				{
					int x			= tid % frameSize[0];
					int y			= tid / frameSize[0];
					bool inASRegion = (x < ex && x >= bx && y < ey && y >= by);
					if (inASRegion) {
						int spp = mAdaptiveSamplingSpp;
						// if the 2 region intersect, spp = spp - 1
						if (tid < tASPixels) {
							spp -= 1;
						}
						cameraRayQueue->mThp[rayIdx] = Spectrum(1.0f / spp);
					}
				}
			},
			stream);
	} else {
		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				Sampler sampler		= &mPixelState->mSampler[pixelId];
				Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
				Ray cameraRay		= mCamera->getRay(pixelCoord, frameSize, sampler);
				cameraRayQueue->pushCameraRay(cameraRay, pixelId);
			},
			stream);
	}
}

void NRRSPathTracer::resize(const Vector2i &size) {
	Log(Info, "[NRRSPathTracer] Resizing to %dx%d", size[0], size[1]);
	const int resolution720p = 1280 * 720;
	const int resolutionSize = size[0] * size[1];
	if (resolutionSize > resolution720p) {
		Log(Fatal, "Currently maximum number of pixels is limited to %d", resolution720p);
	}
	RenderPass::resize(size);
	initialize(false); // need to resize the queues

	// [Why?] it will causes error if it is called before initialize()
	const int maxTrainingSamples = resolutionSize * mMaxRateForPathNodesBuffer;
	mNRRSParams.mNet->resize(size, mMaxQueueSize, maxTrainingSamples);

	auto frameSizeScaled = size / mErrorImageScale;
	mDenoiseTask->resize(frameSizeScaled);
}

void NRRSPathTracer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	mDenoiseTask->setScene(scene);

	if (!mBackend) {
		mBackend	= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(NRRS_BB_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false)
						  .addRaygenEntry("ClosestSimple")
						  .addRaygenEntry("ShadowSimple")
						  .addRayType("ClosestSimple", true, true, false)
						  .addRayType("ShadowSimple", false, true, false);

		mBackend->initialize(params);
	}
	mBackend->setScene(scene);
	mLightSampler = mBackend->getSceneData().lightSampler;
	mNRRSParams.mNet->setScene(scene);
}

void NRRSPathTracer::beginFrame(RenderContext *context) {
	if (!mScene || !mMaxQueueSize) {
		return;
	}

	PROFILE("Begin frame");
	const auto stream = gpContext->cudaStream;
	auto frameSize	  = getFrameSize();
	cudaMemcpyAsync(mCamera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
					cudaMemcpyHostToDevice, stream);

	if (mLowPowerMode && !mLPMRender) {
		// if low power mode is on, we reuse the last frame
		return;
	}

	const bool training = !mNRRSParams.mStopTraining;

	// reset weighted L(current)
	auto frameSizeScaled	  = frameSize / mErrorImageScale;
	int frameSizeScaledLength = frameSizeScaled[0] * frameSizeScaled[1];

	if (training) {
		GPUParallelFor(
			frameSizeScaledLength,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				mWeightedLBufferCurrent[pixelId]	= RGB(0, 0, 0);
				mWeightedLBufferCurrentAcc[pixelId] = 0;
				mNumberSamplesThisPatch[pixelId]	= 0; // should before atomicAdd in next call
			},
			stream);
	}

	int resolutionSize = frameSize[0] * frameSize[1];
	GPUParallelFor(
		mMaxQueueSize,
		KRR_DEVICE_LAMBDA(const int pixelId) {
			// reset per-pixel mRadiance & sample state
			Vector2i pixelCoord		 = {pixelId % frameSize[0], pixelId / frameSize[0]};
			mPixelState->mL[pixelId] = 0;
			mPixelState->mTrainingSamples[pixelId] = 0;
			mPixelState->mSampler[pixelId].setPixelSample(pixelCoord, mFrameId * mSamplesPerPixel);
			mPixelState->mSampler[pixelId].advance(256 * pixelId + mRandomOffset);

			if (training) {
				if (pixelId < resolutionSize) {
					int pixelIdScaled =
						offsetFrame2Scaled(pixelId, uint(frameSize[0]), mErrorImageScale);
					// generateCameraRays() => 1
					atomicAdd(mNumberSamplesThisPatch + pixelIdScaled, 1.0f);
				}
			}
		},
		stream);
}

template <bool tIsSimpleMode, bool tIsTraining>
void NRRSPathTracer::generateRRSNumber(const int depth) {
#define getThpArray() (tIsSimpleMode ? mScatterRayQueueSimple->mThp : mScatterRayQueue->mThp)
#define getPixelIdArray()                                                                          \
	(tIsSimpleMode ? mScatterRayQueueSimple->mPixelId : mScatterRayQueue->mPixelId)
#define getPixelIdArrayCPU()                                                                       \
	(tIsSimpleMode ? mScatterRayQueueSimplePixelIdPtr : mScatterRayQueuePixelIdPtr)

#define RECORD_RRS(type)                                                                           \
	(mShowBuffer.showRRS() && (mShowBuffer.mShowRRSMode == type) &&                                \
	 ((1 << depth) & mShowBuffer.mShowRRSWhichDepth))

	PROFILE("RRS Number generation");
	const cudaStream_t &stream			 = gpContext->cudaStream;
	std::shared_ptr<NRRSNetwork> network = mNRRSParams.mNet;

	int inferenceQueueSize = 0;
	inferenceQueueSize =
		*(int *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mInferQueueSizeInCPUBuffer);
	if (!inferenceQueueSize) {
#if 1 // [TODO] For Debug
		cudaStreamSynchronize(stream);
		inferenceQueueSize = mInferenceQueue->size();
		Log(Info, "[Debug][generateRRSNumber] size after Sync: %d", inferenceQueueSize);
#else
		inferenceQueueSize = mMaxQueueSize;
		Log(Info, "[generateRRSNumber] use mMaxQueueSize in nrrs_generate_inference_data()");
#endif
	}

	// if UBS on, mLossUseNetworkOutputRRS has no effects
	const bool useNetworkOutputRRS = (mUBSSearch || mUseBestStrategy)
										 ? UBSUseNetOutputRRS(depth)
										 : mNRRSParams.mLossUseNetworkOutputRRS;

	{
		PROFILE("Data preparation");
		uint *pixelId = getPixelIdArrayCPU();
		network->prepareInferenceData(inferenceQueueSize, mInferenceQueue, mScene->getBoundingBox(),
									  useNetworkOutputRRS, mRenderedImage, pixelId);
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

	// Debug
	if (mShowBuffer.showRRS() && ((1 << depth) & mShowBuffer.mShowRRSWhichDepth)) {
		int specularSize = -1;
		specularSize =
			*(int *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mSpecularTidSizeInCPUBuffer);
		if (specularSize == -1) {
			// sync as this is allocated by cudaMallocManaged()
			Log(Info, "Sync: specular size");
			cudaStreamSynchronize(stream);
			specularSize = mScatterTidQueue->size();
		}
		// if 0, kernel will not be launched
		if (specularSize) {
			// add 1 to mRRSBuffer
			float *showRRSBuffer = mShowBuffer.mShowRRSBuffer;
			GPUParallelFor(
				specularSize,
				KRR_DEVICE_LAMBDA(const int tid) {
					const uint sid	   = mScatterTidQueue->mTid[tid];
					const uint pixelId = getPixelIdArray()[sid];
					atomicAdd(showRRSBuffer + pixelId, 1.0f);
				},
				stream);
		}
	}

	inferenceQueueSize =
		*(int *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mInferQueueSizeInCPUBuffer);
	if (!inferenceQueueSize) {
		// inferenceQueueSize = mMaxQueueSize;
		Log(Info, "[generateRRSNumber] Sync");
		// sync as this is allocated by cudaMallocManaged()
		cudaStreamSynchronize(stream);
		// real size, if not, should check all GPUParallelFor
		inferenceQueueSize = mInferenceQueue->size();
		Log(Info, "[generateRRSNumber] size after Sync: %d", inferenceQueueSize);
	}

	{
		PROFILE("Network inference");

		uint *pixelId = getPixelIdArrayCPU();
		network->inference(inferenceQueueSize, !useNetworkOutputRRS, mInferenceQueue,
						   mRenderedImage, pixelId);
	}

	{
		PROFILE("Gen RRS Number");
		float *rrsArray			   = mNRRSParams.mRRSArray;
		const bool useWeightWindow = mNRRSParams.mUseWeightWindow;

		const bool updateShowRRSBuffer0 = RECORD_RRS(ShowDebugBuffer::BeforeClamp);
		// const bool isStep3				= mNRRSParams.mNet->getStep() == 3;
		const bool updateShowRRSBuffer1 = RECORD_RRS(ShowDebugBuffer::AfterClamp);
		float *showRRSBuffer			= mShowBuffer.mShowRRSBuffer;

		const precision_t *outputDataPtr = useNetworkOutputRRS
											   ? network->getInferenceOutputBufferPtr1()
											   : network->getInferenceOutputBufferPtr_L();

		Vector2i frameSize = getFrameSize();
		Vector2i drawCenter{mDebugPixel % frameSize[0], mDebugPixel / frameSize[0]};

		float *rrsArrayCeil = mNRRSParams.mRRSArrayCeil;
		const bool useCeil	= mRRSNormalizeUseCeil;

		if (!tIsTraining && !mRenderedImageDenoiseOnce &&
			!mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer) {
			// for adn
			denoiseRenderedImage();
		}

		GPUParallelFor(
			inferenceQueueSize, // real size
			KRR_DEVICE_LAMBDA(const int tid) {
				float rrs = 1.0f;

				if (useNetworkOutputRRS) {
					const uint sid	   = mInferenceQueue->mScatterQueueIndex[tid];
					const uint pixelId = getPixelIdArray()[sid];
					// const RGB thp	   = getThpArray()[sid].toRGB({},
					// *KRR_DEFAULT_COLORSPACE_GPU);
					RGB refI = RGB(0.5f);
					if (mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer) {
						refI = mRenderedImage->getPixel(pixelId).head<3>();
					} else {
						uint pixelIdScaled =
							offsetFrame2Scaled(pixelId, frameSize[0], mErrorImageScale);
						refI = mRenderedImageDenoised[pixelIdScaled];
					}

					const float rrsRaw = *(outputDataPtr + tid * NRRS_RRSNET_DIM_OUTPUT_PADDED);
					//  softplus: log(1 + exp(x))
					//  rrs = log1p(exp(rrs));
					// sigmoid: 1 / (1 + exp(-x))
					// if (isStep3) {
					// rrs = 20.0f / (1.0f + exp(-rrs));
					//}

					const float rrsActi = activationSigma(rrsRaw); // activation

					// rrs = rrsActi / (refI.mean() + 0.01f);

					// if (mEnableRRSScaler) {
					// rrs *= mRRSScaler[depth];
					//}

					rrs = rrsActi;

#if 1 // DEBUG
					if (mRRSDivInRect) {
						auto sampler	= &mPixelState->mSampler[tid];
						const float rnd = sampler->get1D();
						if (rnd < 0.4f) {
							int x = pixelId % frameSize[0];
							int y = pixelId / frameSize[0];
							if (abs(x - drawCenter[0]) <= mDrawRectSize &&
								abs(y - drawCenter[1]) <= mDrawRectSize) {
								rrs /= mRRSDivInRectVal;
							}
						}
					}
#endif

					if (mDebugTrainingData && mDebugPixel == pixelId) {
						printf("[depth = %d] rrs_raw = %g, rrsActi: %g, rrs: %g, ref_mean: %g\n",
							   depth, rrsRaw, rrsActi, rrs, refI.mean());
					}

					// rrs = 1.0 / (refI.mean() + 0.01f);

					// rrs = refI.mean();
					//  rrs = clamp(rrs, 0.0, 100.0f);
				} else {
					const uint sid	   = mInferenceQueue->mScatterQueueIndex[tid];
					const uint pixelId = getPixelIdArray()[sid];
					const RGB thp	   = getThpArray()[sid].toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
					const RGB refI	   = mRenderedImage->getPixel(pixelId).head<3>() + 0.01f;

					const precision_t *pred = (outputDataPtr + tid * NRRS_LL2NET_DIM_OUTPUT_PADDED);

					const RGB L = clamp(RGB(pred[0], pred[1], pred[2]), 1e-4f, 1e4f);
					rrs			= (L * thp / RGB(refI)).mean();
					// rrs = (L.mean() * thp.mean() / RGB(refI).mean());
				}

				// rrs = max(rrs, 1e-2f);

				// TODO: guard against NaN
				// if (isnan(rrs)) {
				//	rrs = 1.0f;
				// }

				const float rrsOrignal = rrs;

				if (useWeightWindow) {
					rrs = RRSWeightWindow(rrs);
				}

				// TODO: now for step3, step2 should clamp
				// if (!useNetworkOutputRRS) {
				// rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);
				// }

				// avoid overflow after normalization
				rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);

				if (updateShowRRSBuffer0 || updateShowRRSBuffer1) {
					// very slow, we have got the same result before, however, it is for
					// debugging
					const uint sid	   = mInferenceQueue->mScatterQueueIndex[tid];
					const uint pixelId = getPixelIdArray()[sid];
					const float rrsRes = updateShowRRSBuffer0 ? rrsOrignal : rrs;
					// if just depth=0, there is no need to use atomicAdd
					// showRRSBuffer[pixelId] += rrsRes;
					atomicAdd(showRRSBuffer + pixelId, rrsRes);
				}

				// [Important] why here? if sum < resolutionSize, we don't need to normalize
				// rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);
				rrsArray[tid] = rrs;

				if (useCeil) {
					rrsArrayCeil[tid] = ceilf(rrs);
				}
			},
			stream);
	}

	{
		PROFILE("RRS Number Nomarlization");
		const float *rrsArray =
			mRRSNormalizeUseCeil ? mNRRSParams.mRRSArrayCeil : mNRRSParams.mRRSArray;

		float *sum		 = mNRRSParams.mTempGPUBuffer;
		float *sumRcpPos = sum + mNRRSParams.mSumPosInGPUBuffer;
		float *partSum	 = sum + (32 * 4); // warp size * 4
		calcSum2PassAsync<true>(rrsArray, sumRcpPos, partSum, inferenceQueueSize, stream);

		if (mDebugOn || sShowError) {
			cudaStreamSynchronize(stream);
			// copy back rrsArray to CPU and calculate the sum
			std::vector<float> rrsArrayCPU(inferenceQueueSize);
			cudaMemcpy(rrsArrayCPU.data(), rrsArray, sizeof(float) * inferenceQueueSize,
					   cudaMemcpyDeviceToHost);
			float sumCpuCal = 0;
			for (int i = 0; i < inferenceQueueSize; i++) {
				sumCpuCal += rrsArrayCPU[i];
			}
			Log(Info, "sum(CPU): %.4f", sumCpuCal);

			if (sShowError) {
				// big => small
				float *data = rrsArrayCPU.data();
				std::sort(data, data + inferenceQueueSize, std::greater<float>());
				const int k[] = {0, 10, 50, 100, 200, 500, 1000, 10000, 100000};
				for (int i = 0; i < std::size(k); i++) {
					Log(Info, "rrs[%d]: %f", k[i], rrsArrayCPU[k[i]]);
				}
				for (int i = 0; i < std::size(k); i++) {
					Log(Info, "rrs[%d]: %f", inferenceQueueSize - 1 - k[i],
						rrsArrayCPU[inferenceQueueSize - 1 - k[i]]);
				}
			}

			cudaStreamSynchronize(stream);
			float sumCPU = 0;
			cudaMemcpy(&sumCPU, sumRcpPos, sizeof(float), cudaMemcpyDeviceToHost);
			cudaStreamSynchronize(stream);
			Log(Info, "sum(rcp): %.4f", 1.0f / sumCPU);
		}
	}

	{
		PROFILE("Gen Tids");
		const float *rrsSumRcp	  = mNRRSParams.mTempGPUBuffer + mNRRSParams.mSumPosInGPUBuffer;
		const float *rrsArray	  = mNRRSParams.mRRSArray;
		const uint resolutionSize = getFrameSize()[0] * getFrameSize()[1];
		const uint *specularSize =
			(uint *) (mNRRSParams.mTempGPUBuffer + mNRRSParams.mSpecularTidSizeInGPUBuffer);

		// the real cost
		const bool updateShowRRSBuffer2 = RECORD_RRS(ShowDebugBuffer::Normalized);
		const bool updateShowRRSBuffer3 = RECORD_RRS(ShowDebugBuffer::RealRay);

		const bool copyTheRRSNode = mCopyTheRRSNode;

		float *showRRSBuffer = mShowBuffer.mShowRRSBuffer;

		const bool shouldScaleUp = !mDonnotNormalizeWhenLessThanOne;

		// [TODO] debug
		const bool printNormalizeFactor = mDebugPixel >= 0 && mDebugPixel == 31565;

		const bool useCeil = mRRSNormalizeUseCeil;

		GPUParallelFor(
			inferenceQueueSize, // real size
			KRR_DEVICE_LAMBDA(const int tid) {
				// here: tid and stid is one to one mapping
				const uint stid = mInferenceQueue->mScatterQueueIndex[tid];
				const RGB thp	= getThpArray()[stid].toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

				// use ceil
				const float normalizeFactor = ((resolutionSize - *specularSize) * (*rrsSumRcp)) *
											  (useCeil ? 1.0f : RRS_NORMALIZE_RATE);

				if (printNormalizeFactor && tid == 0) {
					printf("[Debug] depth = %d, normalizeFactor: %f\n", depth, normalizeFactor);
				}

				// Debug
				if (mDebugRRSNormalized) {
					if (tid == 0) {
						printf("depth: %d, normalized: %d\n", depth,
							   shouldScaleUp || (normalizeFactor < 1.0f));
					}
				}

				float rrs = rrsArray[tid];
				if (shouldScaleUp) {
					rrs = rrs * normalizeFactor;
				} else {
					// if sum < resolution, don't normalize
					rrs = rrs * (normalizeFactor < 1.0f ? normalizeFactor : 1.0f);
				}

				// rrs = clamp(rrs, RRS_CLAMP_MIN, RRS_CLAMP_MAX);

				if (tIsSimpleMode) {
					getThpArray()[stid] = thp / rrs; // thpNew
				} else {
					// now update the mThp in the handleIntersections()
					// getThpArray()[stid] = thp / rrs; // thpNew

					// record RRS after normalization
					mScatterRayQueue->mRRS[stid] = rrs;
				}

				auto sampler	= &mPixelState->mSampler[tid];
				const float rnd = sampler->get1D();
				// rrsArray[tid]	= rrs; // update rrs

				int s			  = int(rrs);
				const float sLeft = rrs - s;

				if (!copyTheRRSNode && (!tIsSimpleMode && tIsTraining)) {
					const bool noRays = (s == 0) && (rnd > sLeft);
					if (!noRays) {
						// update mNodeIdx
						// [1] tIsSimpleMode = false;
						// [2] specular won't come here

						const int pixelId = mScatterRayQueue->mPixelId[stid];
						int nodeIdxOld	  = mScatterRayQueue->mNodeIdx[stid];

						const auto intrSoa = mScatterRayQueue->mIntr;

						int nodeIdx = mPathState->recordNode(
							nodeIdxOld, intrSoa.p[stid], intrSoa.wo[stid],
							intrSoa.sd.roughness[stid], thp, pixelId, rrs, depth);
						if (nodeIdx == -1) {
							nodeIdx = nodeIdxOld;
						}
						mScatterRayQueue->mNodeIdx[stid] = nodeIdx;
					}
				}

				if (updateShowRRSBuffer2 || updateShowRRSBuffer3) {
					const uint pixelId = getPixelIdArray()[stid];
					float rrsRes	   = updateShowRRSBuffer2 ? rrs : float(s + (rnd <= sLeft));
					// showRRSBuffer[pixelId] += rrsRes;
					atomicAdd(showRRSBuffer + pixelId, rrsRes);
				}
#if 0
				if (s > 0) {
					mScatterTidQueue->push(stid, s);
				}
				if (rnd <= sLeft) {
					mScatterTidQueue->push(stid);
				}
#else // faster but worse?
				s += (rnd <= sLeft) ? 1 : 0;
				if (s > 0) {
					mScatterTidQueue->push(stid, s);
				}
#endif
			},
			stream);

		// cudaStreamSynchronize(stream);
		// auto scatterTidSize = mScatterTidQueue->size();
		// if (scatterTidSize > mMaxQueueSize) {
		//	mDebugOn = true;
		// }

		if (mDebugOn) {
			cudaStreamSynchronize(stream);
			auto scatterTidSize = mScatterTidQueue->size();
			if (scatterTidSize > mMaxQueueSize) {
				Log(Warning, "%d > %d", scatterTidSize, mMaxQueueSize);
			}
			Log(Info, "[depth = %d]: r = %d / %d = %.4f, infer: %d, r_max = %.4f", depth,
				scatterTidSize, resolutionSize, 1.0f * scatterTidSize / resolutionSize,
				inferenceQueueSize, 1.0f * scatterTidSize / mMaxQueueSize);
		}
	}
#undef RECORD_RRS
#undef getThpArray
#undef getPixelIdArray
#undef getPixelIdArrayCPU
}

template <bool tIsTraining> void NRRSPathTracer::renderInternal(RenderContext *context) {
	PROFILE("NRRS Path Tracer");
	const auto frameSize	  = getFrameSize();
	const auto stream		  = gpContext->cudaStream;
	const uint resolutionSize = frameSize[0] * frameSize[1];

	mShowBuffer.mShowRRSAcc += mShowBuffer.showRRS() ? 1 : 0;
	mShowBuffer.mShowSpecularAcc += mShowBuffer.showSpecular() ? 1 : 0;

	mSamplesPerPixelThisFrame = 0;
	for (int sampleId = 0; sampleId < mSamplesPerPixel; sampleId++) {
		mSamplesPerPixelThisFrame++;
		// [STEP#0] reset training environment
		GPUCall(
			KRR_DEVICE_LAMBDA() {
				currentRayQueue(0)->reset();
				if (tIsTraining) {
					mPathState->reset(mDebugOn);
					mTrainBuffer->clear();
				}
			},
			stream);

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

			const bool queryNetworkThenBreak =
				(mShowBuffer.showLi() && (depth >= mShowBuffer.mShowLiDepth));
			const bool shouldInfer = isInferMode(depth) || queryNetworkThenBreak;

			// [STEP#2.1] find closest intersections, filling in mScatterRayQueue and
			// hitLightQueue
			traceClosest(depth);

			if (shouldInfer) {
				// get inferenceQueueSize
				uint *p =
					(uint *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mInferQueueSizeInCPUBuffer);
				*p = 0;
				cudaMemcpyAsync(p, ((byte *) mInferenceQueue) + mInferenceQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToHost, stream);

				if (mShowBuffer.showRRS()) {
					// specular queue size(CPU)
					p  = (uint *) (mNRRSParams.mTempCPUBuffer +
								   mNRRSParams.mSpecularTidSizeInCPUBuffer);
					*p = 0;
					cudaMemcpyAsync(p,
									((byte *) mScatterTidQueue) + mScatterTidQueue->offsetOfSize(),
									sizeof(uint), cudaMemcpyDeviceToHost, stream);
				}

				// specular queue size(GPU)
				p = (uint *) (mNRRSParams.mTempGPUBuffer + mNRRSParams.mSpecularTidSizeInGPUBuffer);
				cudaMemcpyAsync(p, ((byte *) mScatterTidQueue) + mScatterTidQueue->offsetOfSize(),
								sizeof(uint), cudaMemcpyDeviceToDevice, stream);
			}

			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleEmissiveHit<tIsTraining>();
			handleMiss<tIsTraining>();

			if (queryNetworkThenBreak) {
				if (mShowBuffer.mShowLiAddRadiance) {
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
				generateRRSNumber<false, tIsTraining>(depth);
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

		if (tIsTraining && mFrameId) {
			// get pathState->size()
			uint *p2psSize =
				(uint *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mPathStateSizeInCPUBuffer);
			*p2psSize = 0;
			cudaMemcpyAsync(p2psSize, ((byte *) mPathState) + mPathState->offsetOfSize(),
							sizeof(uint), cudaMemcpyDeviceToHost, stream);

			// ignore first frame to make sure mPixelState->mAccSppForError > 0
			trainStep();
		}
	}

	if (mAutoTrainRequestExit) {
		mShowBuffer.mShowType		 = ShowDebugBuffer::ShowType::ShowLError;
		mShowBuffer.mShowScalarJetOn = false;
	}

	// UBS
	if (mUBSSearch) {
		UBSAnalyseFrame();
	}

	// write results of the current frame...
	writeResultToRenderTarget(context);

	// debug
	if (mShowBuffer.mCalcAvg || mAutoTrainRequestExit) {
		float *sumPtr		= mShowBuffer.mSumGPUPtr;
		mShowBuffer.mSumCPU = 0;

		// reset
		GPUCall(KRR_DEVICE_LAMBDA() { *sumPtr = 0; }, stream);

		CudaRenderTarget cudaFrame = context->getColorTexture()->getCudaRenderTarget();

		// calculate
		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				float res = cudaFrame.read(pixelId).head<3>().mean();
				atomicAdd(sumPtr, res);
			},
			stream);

		// copy back
		cudaMemcpyAsync(&mShowBuffer.mSumCPU, sumPtr, sizeof(float), cudaMemcpyDeviceToHost,
						stream);

		if (mAutoTrainRequestExit) {
			saveAndExit();
		}
	}
}

void NRRSPathTracer::controlExp(RenderContext *context) {
#define EXP_DELTA_TIME(x) (CpuTimer::calcDuration(x, CpuTimer::getCurrentTimePoint()) * 1e-3)
#define EXP_FINISHED                                                                               \
	{                                                                                              \
		mNRRSParams.mStopTraining	= true;                                                        \
		mNRRSParams.mSimpleRenderOn = true;                                                        \
		changeRenderMode();                                                                        \
		beginFrame(context);                                                                       \
		Log(Info, "[Exp] Start Inference for %g seconds", mExpInferenceTime);                      \
		mExpStartTime = CpuTimer::getCurrentTimePoint();                                           \
		mExpState	  = 2;                                                                         \
	}

	static int sTrainState = -1;

	if (mExpState == 0) {
		Log(Info, "[Exp] Start Train for %g seconds", mExpTrainTime);
		mExpStartTime			  = CpuTimer::getCurrentTimePoint();
		mNRRSParams.mStopTraining = false;

		mExpState = 1;

		sTrainState							 = 0;
		mEnableRRS							 = false;
		mNRRSParams.mLossUseNetworkOutputRRS = false;
		mNRRSParams.mNet->setStep(1);

		// TODO: for adn, only L is needed
		if (mExpMethods == Exp_ADN) {
			mNRRSParams.mNet->setStep(0); // only train L not L2
			mNRRSParams.mNet->setShowLossIndexLL2();
		}

		if (mExpMethods == ExpMethods::Exp_NRRS_MIX) {
			mExpTrainTime -= 25.0f; // left for LL2Net & no denoise & Mix-Depth
		} else if (mExpMethods == ExpMethods::Exp_NRRS) {
			mExpTrainTime -= 15.0f; // left for LL2Net & no denoise
		}

		if (mExpDenoiseAlways) {
			mExpTrainTime += 5.0f;
		}

	} else if (mExpState == 1) {
		// train

		{ // control in detail

			if (sTrainState == 0) {
				// train LL2Net
				float t1 = 10.0f;
				if (mExpMethods == ExpMethods::Exp_ADN) {
					t1 = mExpTrainTime;
				}
				if (EXP_DELTA_TIME(mExpStartTime) >= t1) {
					mEnableRRS							 = true;
					mNRRSParams.mLossUseNetworkOutputRRS = true;
					mNRRSParams.mNet->setStep(3);
					mDenoiseTask->setEnabled(true);

					mExpStartTime = CpuTimer::getCurrentTimePoint();
					sTrainState	  = 1;

					if (mExpMethods == ExpMethods::Exp_ADN) {
						mNRRSParams.mLossUseNetworkOutputRRS = false;
						// finished
						EXP_FINISHED;
					}
				}
			} else if (sTrainState == 1) {
				// train RRSNet with denoise
				if (EXP_DELTA_TIME(mExpStartTime) >= mExpTrainTime) {
					if (!mExpDenoiseAlways) {
						mDenoiseTask->setEnabled(false);
					}

					mExpStartTime = CpuTimer::getCurrentTimePoint();
					sTrainState	  = mExpDenoiseAlways ? 3 : 2;
				}
			} else if (sTrainState == 2) {
				// train RRSNet without denoise
				if (EXP_DELTA_TIME(mExpStartTime) >= 5.0f) {
					sTrainState = 3;

					mNRRSParams.mStopTraining = true;
					if (mExpMethods == ExpMethods::Exp_NRRS_MIX) {
						UBSResetAndBeginSearch();
					} else if (mExpMethods == ExpMethods::Exp_NRRS) {
						// finished
						EXP_FINISHED;
					}
				}
			} else if (sTrainState == 3) {
				// Mix-Depth ：~10s
				if (!mUBSSearch) {
					mUseBestStrategy = true;

					EXP_FINISHED;
				}
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

#undef EXP_FINISHED
#undef EXP_DELTA_TIME
}

void NRRSPathTracer::render(RenderContext *context) {

	if (mAutoTrain) {
		autoTrainControl();
	}

	if (!mScene || !mMaxQueueSize) {
		return;
	}

	if (mUBSSearch) {
		if (!UBSNextState()) {
			UBSEndSearch();
		}
	}

	if (mExpOn) {
		controlExp(context);
	}

	// update the mRenderedImage if the scene/camera changes
	if (mScene->getChanges()) {
		// camera changes will enter this branch
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
		mDenoiseTask->resetState();
		mDenoiseTask->setGBufferInvalid();
	}

	static size_t lastResetFrame = 0;
	auto lastSceneUpdates		 = mScene->getSceneGraph()->getLastUpdateRecord();
	if (lastSceneUpdates.updateFlags != SceneGraphNode::UpdateFlags::None &&
		lastResetFrame < lastSceneUpdates.frameIndex) {
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
		lastResetFrame = lastSceneUpdates.frameIndex;
	}

	// low power mode
	if (mLowPowerMode) {
		if (!mLPMRender) {
			writeResultToRenderTarget(context);
			Sleep(1000 / 60);
			return;
		}
	}

	if (mNRRSParams.mSimpleRenderOn) {
		renderSimple(context);
	} else {
		const bool stopTraining		 = mNRRSParams.mStopTraining;
		const bool trainingAndShowLi = !stopTraining && mShowBuffer.showLi();

		if (trainingAndShowLi) {
			mShowBuffer.mShowType = ShowDebugBuffer::ShowType::None;
		}

		if (stopTraining) {
			renderInternal<false>(context);
		} else {
			renderInternal<true>(context);
		}

		if (trainingAndShowLi) {
			beginFrame(context);
			mShowBuffer.mShowType = ShowDebugBuffer::ShowType::ShowLi;
			renderInternal<false>(context);
		}
	}
}

void NRRSPathTracer::endFrame(RenderContext *context) { mFrameId++; }

void NRRSPathTracer::finalize() {
	// TODO: free all the memory
	if (mCamera) {
		cudaFree(mCamera);
	}
	if (mNRRSParams.mRRSArray) {
		cudaFree(mNRRSParams.mRRSArray);
	}
	if (mNRRSParams.mRRSArrayCeil) {
		cudaFree(mNRRSParams.mRRSArrayCeil);
	}
	if (mNRRSParams.mTempGPUBuffer) {
		cudaFree(mNRRSParams.mTempGPUBuffer);
	}
	if (mNRRSParams.mTempCPUBuffer) {
		delete[] mNRRSParams.mTempCPUBuffer;
	}

	if (mRenderedImageDenoised) {
		cudaFree(mRenderedImageDenoised);
	}

	// if (mCostBuffer) {
	// cudaFree(mCostBuffer);
	//}

	mShowBuffer.freeBuffer();

	if (mUBSArray) {
		delete[] mUBSArray;
	}
	if (mUBSArrayChar) {
		delete[] mUBSArrayChar;
	}

	if (mUBSErrorBufferGPU) {
		cudaFree(mUBSErrorBufferGPU);
	}

	if (mExpRayCounter) {
		cudaFree(mExpRayCounter);
	}
}

void NRRSPathTracer::ShowDebugBuffer::renderUI(NRRSPathTracer *pass) {
	ui::PushID("Show Debug Buffer");

	const static char *sShowTypes[]	  = {"Li",
										 "Weighted L",
										 "Rendered Image",
										 "Rendered Image (Denoised)",
										 "Ref Image",
										 "RRS",
										 "Specular",
										 "L Error",
										 "Weighted L Error",
										 "Error Per Pixel[training]",
										 "Training Samples Per Pixel[discard]",
										 "Number Samples Per Pixel[all]",
										 "Network Debug",
										 "Error Per Sample[training]",
										 "None"};
	const static char *sShowLiTypes[] = {"Li", "sqrt(Li^2)", "sqrt(var)"};
	const auto resolutionSize		  = pass->getFrameSize()[0] * pass->getFrameSize()[1];

	if (ui::TreeNodeEx("Show Debug Buffer", ImGuiTreeNodeFlags_DefaultOpen)) {

#define BB_Button(type)                                                                            \
	if (ui::Button(sShowTypes[(int) type])) {                                                      \
		mShowType = type;                                                                          \
	}
		BB_Button(ShowType::None);
		ui::SameLine();
		BB_Button(ShowType::ShowRRS);
		ui::SameLine();
		BB_Button(ShowType::ShowLi);
		ui::SameLine();
		BB_Button(ShowType::ShowNetworkDebug);
#undef BB_Button
		auto stream = gpContext->cudaStream;
		ui::Combo("Show Type", (int *) &mShowType, sShowTypes, std::size(sShowTypes));
		switch (mShowType) {
			case ShowLi: {
				ui::Combo("Show Li Type", (int *) &mShowLiType, sShowLiTypes,
						  std::size(sShowLiTypes));
				ui::Checkbox("Add Li Radiance", &mShowLiAddRadiance);
				ui::Checkbox("Gray Scale", &mShowLiGray);
				ui::SliderInt("Min Depth to Query Network", &mShowLiDepth, 0, 10);
			} break;
			case ShowRRS: {
				bool resetRRSBuffer				  = false;
				static const char *sShowRRSMode[] = {"Before Clamp", "After Clamp", "Normalized",
													 "Real Ray"};
				resetRRSBuffer |= ui::Combo("Show RRS Mode", (int *) &mShowRRSMode, sShowRRSMode,
											int(ShowRRSMode::Count));
				{ // depth hint
					static char sShowRRSHint[32]   = "100000";
					static bool sFirstConstructStr = true;
					static std::string sShowStr{};

					if (ui::InputText("Show RRS Depth Hint", sShowRRSHint, 64) ||
						sFirstConstructStr) {
						sFirstConstructStr = false;
						sShowStr		   = "Show RRS At ";

						uint showWhichDepth = pass->loadArrayToInt(sShowRRSHint, sShowStr);
						mShowRRSWhichDepth	= showWhichDepth;
					}

					ui::Text("%s", sShowStr.c_str());
				}

				ui::Checkbox("Reset RRS Buffer Each Frame", &mResetRRSBufferEachFrame);
				resetRRSBuffer |= ui::Button("Reset RRS Buffer");
				if (!pass->mLowPowerMode && (mResetRRSBufferEachFrame || resetRRSBuffer)) {
					cudaMemsetAsync(mShowRRSBuffer, 0, sizeof(float) * resolutionSize, stream);
					mShowRRSAcc = 0;
				}
				ui::Text("[RRS] Accmulate: %d", mShowRRSAcc);
			} break;
			case ShowSpecular: {
				if (ui::Button("Reset Specular Buffer")) {
					cudaMemsetAsync(mShowBuffer, 0, sizeof(float) * resolutionSize, stream);
					mShowSpecularAcc = 0;
				}
				ui::Text("[Specular] Accmulate: %d", mShowSpecularAcc);
			} break;
			case ShowWeightedLError: {
				ui::Text("check clamp int the loss module!");
			} break;
			case ShowErrorPerPixel: {
				auto &params = mShowErrorPerPixelParams;
				ui::Checkbox("Log Scale", &params.mLogScale);
				ui::Checkbox("Only Show Larger Than", &params.mLargeThan);
				if (params.mLargeThan) {
					ui::SliderFloat("  ErrorPerPixel Bound", &params.mLargeBound, 0.0f, 100.0f);
				} else {
					ui::Checkbox("Show Relative to Mean", &params.mShowRelativeToMean);
					ui::SliderFloat("  Show Relative to Mean Scale",
									&params.mShowRelativeToMeanScale, 0.5f, 30.f);
				}
			} break;
			case ShowLError:
			case ShowWeightedL:
				// mWeightedLBlendWeight is not set here, as it also effects the training
			case ShowRenderedImage:
			case ShowTrainingSamplesPerPixel:
			case ShowNumberSamplesPerPixel:
			case ShowNetworkDebug: {
				ui::Checkbox("Type, true is float, false is uint32_t", &mShowNetDebugTypeIsFloat);
			}
			case ShowRefImage:
			case None:
			default:
				break;
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

		// sum
		ui::Checkbox("Calc Avg", &mCalcAvg);
		if (mCalcAvg) {
			ui::SameLine();
			ui::Text(": %f (%f)", mSumCPU / resolutionSize, mSumCPU);
		}

		ui::TreePop();
	}

	ui::PopID();
}

void NRRSPathTracer::ShowDebugBuffer::allocateBuffer(const uint resolutionSize) {
	const uint size = sizeof(float) * resolutionSize;

	if (mSumGPUPtr) {
		cudaFree(mSumGPUPtr);
	}
	cudaMalloc(&mSumGPUPtr, sizeof(float));

	// rrs
	if (mShowRRSBuffer) {
		cudaFree(mShowRRSBuffer);
	}
	cudaMalloc(&mShowRRSBuffer, size);
	cudaMemset(mShowRRSBuffer, 0, size);
	mShowRRSAcc = 0;

	// specular rays
	if (mShowBuffer) {
		cudaFree(mShowBuffer);
	}
	cudaMalloc(&mShowBuffer, size);
	cudaMemset(mShowBuffer, 0, size);
	mShowSpecularAcc = 0;
}

void NRRSPathTracer::ShowDebugBuffer::freeBuffer() {
	if (mSumGPUPtr) {
		cudaFree(mSumGPUPtr);
	}
	if (mShowRRSBuffer) {
		cudaFree(mShowRRSBuffer);
	}
	if (mShowBuffer) {
		cudaFree(mShowBuffer);
	}
}

void NRRSPathTracer::NRRSParams::renderUI(NRRSPathTracer *pass) {
	// red color button
	sShowError = ui::ColorButton("showError", ImVec4(1, 0, 0, 1));
	mNet->renderUI();

	if (mTrainOneStep || mTrainOneBatch) {
		mTrainOneBatch = false;
		mTrainOneStep  = false;
		mStopTraining  = true;
	}
	ui::Checkbox("Stop Training", &mStopTraining);
	if (mStopTraining) {
		if (ui::Button("Save Network Weight[Very Slow]")) {
			mNet->saveWeights(File::outputDir().string());
		}
		if (ui::Checkbox("Simple Render Mode", &mSimpleRenderOn)) {
			pass->changeRenderMode();
		}
		ui::Checkbox("Use Weight Window", &mUseWeightWindow);
		if (ui::Button("Train One Step")) {
			mTrainOneStep = true;
			mStopTraining = false;
		}
		if (ui::Button("Train One Batch")) {
			mTrainOneBatch = true;
			mStopTraining  = false;
		}
	} else {
		// TODO: guidedState can not hold on so many rays
		if (mSimpleRenderOn) {
			mSimpleRenderOn = false;
			pass->changeRenderMode();
		}
	}

	if (ui::TreeNode("Advanced training options")) {
		if (ui::InputInt("Batch per frame", (int *) &mBatchPerFrame, 1, 1)) {
			mBatchPerFrame = max(0U, mBatchPerFrame);
		}
		ui::SliderFloat("Weighted L Blend Weight[new]", &pass->mWeightedLBlendWeight, 0.0f, 1.0f);
		ui::TreePop();
	}
	ui::Checkbox("Copy the RRS Node", &(pass->mCopyTheRRSNode));
	ui::Checkbox("Ignore Same Training Data For RRSNet", &(pass->mIgnoreSameTrainingDataForRRSNet));
	ui::Checkbox("Ignore Training Data With L = 0", &mIgnoreZeroL);
	ui::Text("Current step: %d; %d samples; loss: %f", sNumLossSamples, sNumTrainingSamples,
			 sCurLossScalar.emaVal());
	ui::Text("batches: min(%d, %d)", sNumTrainingSamples / mBatchSize + 1, mBatchPerFrame);
	ui::PlotLines("Loss graph", sLossGraph.data(), min(sNumLossSamples, sLossGraph.size()),
				  sNumLossSamples < LOSS_GRAPH_SIZE ? 0 : sNumLossSamples % LOSS_GRAPH_SIZE, 0,
				  FLT_MAX, FLT_MAX, ImVec2(0, 50));
}

void NRRSPathTracer::renderUI() {
	int pushid = 7135;
	ui::Text("Frame ID: %d", mFrameId);
	ui::Text("Render parameters");
	mSamplesPerPixel = 1;
	ui::Text("Samples per pixel: %d", mSamplesPerPixel);
	// ui::InputInt("Samples per pixel", &mSamplesPerPixel);
	// ui::InputInt("Max bounces", &mMaxDepth, 1);
	ui::Text("Max Bounces: %d", mMaxDepth);
	ui::SliderFloat("Russian roulette", &mFixedProbRR, 0, 1);
	ui::Checkbox("Enable NEE", &mEnableNEE);
	static float sRRSMax = 100.0f;
	ui::SliderFloat("RRS UpBound", &sRRSMax, 20.0f, 1000.0f);
	ui::SliderFloat("RRS Max", &RRS_CLAMP_MAX, 1.0f, sRRSMax);
	ui::SliderFloat("RRS Min", &RRS_CLAMP_MIN, 0.05f, 1.0f);
	ui::Checkbox("RRS Normalize with Ceil(rrs)", &mRRSNormalizeUseCeil);
	if (!mRRSNormalizeUseCeil) {
		ui::SliderFloat("RRS Normalize Rate", &RRS_NORMALIZE_RATE, 0.1f, 0.95f);
	}
	ui::Checkbox("Donnot Normalize When Less Than One", &mDonnotNormalizeWhenLessThanOne);
	mDebugRRSNormalized = ui::Button("Show Whether RRS is Normalized");

	if (ui::TreeNodeEx("Adaptive Sampling Test")) {
		ui::Checkbox("Adaptive Sampling On", &mAdaptiveSamplingOn);
		ui::SliderInt("Adaptive Sampling Radius", &mAdaptiveSamplingRadius, 10, 200);
		ui::SliderInt("Adaptive Sampling Spp", &mAdaptiveSamplingSpp, 2, 50);

		{
			const int resolutionSize = getFrameSize()[0] * getFrameSize()[1];
			int size				 = mAdaptiveSamplingRadius * 2 + 1;
			size *= size;
			mAdaptiveSamplingSpp = min(mAdaptiveSamplingSpp, resolutionSize / size);
		}

		ui::TreePop();
	}

	if (ui::TreeNodeEx("Low Power Mode Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
		ui::Checkbox("Low Power Mode On", &mLowPowerMode);
		if (mLowPowerMode) {
			mLPMRender = ui::Button("Render One Frame");
			if (mLPMRender) {
				// just as a output segmentation line
				Log(Info, "\n\nRender one frame\n\n");
			}
		}
		ui::TreePop();
	}

	ui::Checkbox("Use AccBuffer(false: Denoised Buffer)",
				 &mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer);
	if (ui::Button("Reset rendered image")) {
		cudaStreamSynchronize(gpContext->cudaStream);
		mRenderedImage->reset();
	}
	if (ui::Button("Denoise rendered image")) {
		denoiseRenderedImage();
	}

	ui::Checkbox("Enable RRS", &mEnableRRS);
	if (mEnableRRS) {
		ui::Checkbox("Use Network Output RRS", &(mNRRSParams.mLossUseNetworkOutputRRS));
		ui::SliderInt("Max RRS Depth(Included)", &mMaxRRSDepthIncluded, 0, 10);

		// ui::Checkbox("Enable RRS Scaler", &mEnableRRSScaler);

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

	ui::Text("NRRSParams");
	mNRRSParams.renderUI(this);
	mShowBuffer.renderUI(this);
	{ // guard
		// if (!mNRRSParams.mStopTraining) {
		//	if (mShowBuffer.showLi()) {
		//		// when training, we can not show Li
		//		mShowBuffer.mShowType = ShowDebugBuffer::None;
		//	}
		// }
		if (!mEnableRRS) {
			if (mShowBuffer.mShowType == ShowDebugBuffer::ShowRRS ||
				mShowBuffer.mShowType == ShowDebugBuffer::ShowSpecular) {
				mShowBuffer.mShowType = ShowDebugBuffer::None;
			}
		}
	}

	if (ui::TreeNodeEx("Denoise")) {
		ui::PushID("NRRSPathTracer::Denoise");
		const static char *sDenoiseInputType[] = {"Error", "L"};
		ui::Combo("Denoise Input Type", (int *) &mDenoiseInputType, sDenoiseInputType,
				  std::size(sDenoiseInputType));
		mDenoiseTask->renderUI(!mShowBuffer.isNone());
		ui::PopID();
		ui::TreePop();
	}

	ui::Text("Debugging");
	ui::Checkbox("Debug On", &mDebugOn);
	ui::InputInt("DebugInt", &mDebugInt);
	ui::Checkbox("Debug output", &mDebugOutput);
	if (mDebugOutput) {
		bool debugPixelChanged = false;
		// left key click, and mouse is not in the ui
		auto &io = ui::GetIO();
		if (!io.WantCaptureMouse && ImGui::IsMouseClicked(0)) {
			auto frame = getFrameSize();
			auto posx  = io.MousePos.x;
			auto posy  = io.MousePos.y;
			// flip posy
			posy = frame[1] - 1 - posy;

			int debugPixelNew = (int) (posy * frame[0] + posx);
			debugPixelChanged = (debugPixelNew != mDebugPixel);
			mDebugPixel		  = debugPixelNew;
		}
		ui::SameLine();
		debugPixelChanged |= ui::InputInt("Debug pixel:", (int *) &mDebugPixel);
		debugPixelChanged |= ui::Checkbox("Debug Training Data", &mDebugTrainingData);
		if (debugPixelChanged) {
			mNRRSParams.mNet->setDebugPixelForTraining(mDebugTrainingData ? mDebugPixel : -1);
		}
	}
	ui::Checkbox("RRS Div in Rect", &mRRSDivInRect);
	ui::SliderFloat("RRS Div in Rect Value", &mRRSDivInRectVal, 0.1f, 10.0f);
	ui::Checkbox("Draw Rect", &mDrawRect);
	ui::SliderInt("Draw Rect Size", &mDrawRectSize, 1, 100);

	if (ui::Button("Reset parameters")) {
		resetTraining();
	}
	ui::Checkbox("Clamping pixel value", &mEnableClamp);
	if (mEnableClamp) {
		ui::SameLine();
		ui::DragFloat("Max:", &mClampMax, 1, 1, 1e5f, "%.1f");
	}
}

void NRRSPathTracer::resetTraining() {
	auto frameSize	   = getFrameSize();
	int resolutionSize = frameSize[0] * frameSize[1];
	mNRRSParams.mNet->reset(resolutionSize, 0, 0);

	// reurn to the initial state
	std::fill(sLossGraph.begin(), sLossGraph.end(), 0);
	sNumLossSamples		= 0;
	sNumTrainingSamples = 0;
	sCurLossScalar		= Ema{Ema::Type::Time, 50};
}

void NRRSPathTracer::denoiseRenderedImage() {
	if (mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer) {
		return;
	}

	const bool enabled	  = mDenoiseTask->isEnabled();
	const bool needAlbedo = mDenoiseTask->getDontNeedAlbedo();

	mDenoiseTask->setEnabled(enabled);
	mDenoiseTask->setDontNeedAlbedo(false);

	Vector2i frameSizeScaled = getFrameSize() / mErrorImageScale;
	const auto stream		 = gpContext->cudaStream;

	mDenoiseTask->renderGBuffer();
	mDenoiseTask->resetState();

	// upload color
	RGB *color = mDenoiseTask->getBuffer(DenoiseTask::Color);
	GPUParallelFor(
		frameSizeScaled[0] * frameSizeScaled[1],
		KRR_DEVICE_LAMBDA(const int pixelId) {
			color[pixelId] = mRenderedImage->getPixelNoTransform(pixelId).head<3>();
		},
		stream);
	mDenoiseTask->denoise(false, mRenderedImageDenoised); // denoise & download color

	// restore
	mDenoiseTask->setDontNeedAlbedo(needAlbedo);
	mDenoiseTask->setEnabled(enabled);

	mRenderedImageDenoiseOnce = true;
}

void NRRSPathTracer::trainStep() {

	PROFILE("Training");
	const cudaStream_t &stream			  = gpContext->cudaStream;
	const auto frameSize				  = getFrameSize();
	const uint resolutionSize			  = frameSize[0] * frameSize[1];
	std::shared_ptr<NRRSNetwork> &network = mNRRSParams.mNet;
	if (!network) {
		logFatal("Network not initialized!");
	}

	{
		PROFILE("Data preparation");
		float *thp, *errorForTraining, *errorForAvgForTraining, *refMean, *sampleWeight,
			*errorPerPixel, *numSamples;
		uint *pixelIDArray;
		network->getLossHelper(thp, errorForTraining, errorForAvgForTraining, refMean, sampleWeight,
							   errorPerPixel, numSamples, pixelIDArray);

		uint frameSizeWidth = uint(frameSize[0]);

		if (mFirstFrameUpdateWeightedL) {
			// first frame, reset pixel state
			mFirstFrameUpdateWeightedL = false;
			GPUParallelFor(
				resolutionSize,
				KRR_DEVICE_LAMBDA(const int pixelId) {
					Spectrum LSpec = mPixelState->mL[pixelId] / mSamplesPerPixelThisFrame;
					RGB L		   = LSpec.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

					int pixelIdScaled =
						offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);

					float *data = (float *) (mWeightedLBuffer + pixelIdScaled);
					atomicAdd(data, L[0]);
					atomicAdd(data + 1, L[1]);
					atomicAdd(data + 2, L[2]);
				},
				stream);
		}

		// [STEP#1] update weighted L
		// [STEP#1.1] update weighted L Current
		GPUParallelFor(
			resolutionSize,
			KRR_DEVICE_LAMBDA(const int pixelId) {
				Spectrum LSpec = mPixelState->mL[pixelId];
				RGB L		   = LSpec.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

				int pixelIdScaled = offsetFrame2Scaled(pixelId, frameSizeWidth, mErrorImageScale);
				float *data		  = (float *) (mWeightedLBufferCurrent + pixelIdScaled);

				atomicAdd(data, L[0]);
				atomicAdd(data + 1, L[1]);
				atomicAdd(data + 2, L[2]);
				atomicAdd(mWeightedLBufferCurrentAcc + pixelIdScaled,
						  uint(mSamplesPerPixelThisFrame));
			},
			stream);

		Vector2i frameSizeScaled  = frameSize / mErrorImageScale;
		int frameSizeScaledLength = frameSizeScaled[0] * frameSizeScaled[1];

		// [STEP#1.2] update weighted L
		LinearKernel(nrrs_update_weighted_L, stream, frameSizeScaledLength, mPixelState,
					 mWeightedLBufferCurrent, mWeightedLBufferCurrentAcc, mWeightedLBuffer,
					 mWeightedLBlendWeight);

		const bool isStep3 = network->getStep() == 3;
		if (isStep3) {
			// error is only needed in step3, however, we need warmup before

			const bool clampPixelError		= network->getClampPixelError();
			const bool errorMultiplySamples = network->getPixelErrorMultiplySamples();

			if (!mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer) {
				denoiseRenderedImage();
			}

			const bool denoisedOn = mDenoiseTask->isEnabled();

			// calculate error for each pixel
			// TODO: clamp should be done in the kernel
			// TODO[IM]: should use mRenderedImage instead of mRefImageDebug
			auto tFunCalcError = [=]() {
				LinearKernel(nrrs_calculate_error, stream, frameSizeScaledLength, errorPerPixel,
							 // mRefImageDebug, // debug only
							 mRenderedImage, mRenderedImageDenoised,
							 mUseTheAccmulatedBufferInsteadOfTheDenosiedBuffer, mPixelState,
							 mWeightedLBuffer, clampPixelError, errorMultiplySamples,
							 mNumberSamplesThisPatch, resolutionSize, mErrorFactorXX,
							 mErrorFactorXY, denoisedOn);
			};

			// denoise
			const bool chooseError = mDenoiseInputType == DenoiseInputType::Error;
			auto tFuncDenoise	   = [=]() {
				 LinearKernel(upload_denoise_error_buffer, stream, frameSizeScaledLength,
								  chooseError, errorPerPixel, mWeightedLBuffer,
								  mDenoiseTask->getBuffer(DenoiseTask::Color));
				 mDenoiseTask->denoise(true);
				 LinearKernel(download_denoise_error_buffer, stream, frameSizeScaledLength,
								  chooseError, errorPerPixel, mWeightedLBuffer,
								  mDenoiseTask->getBuffer(DenoiseTask::DenoisedColor),
								  errorMultiplySamples, mNumberSamplesThisPatch, resolutionSize,
								  mErrorFactorXX, mErrorFactorXY);
			};

			if (denoisedOn) {
				mDenoiseTask->renderGBuffer();
				mDenoiseTask->resetState();
				if (mDenoiseFrames > 0) {
					if (--mDenoiseFrames == 0) {
						mDenoiseTask->setEnabled(false);
						Log(Info, "Denoise task disabled");
					}
				}

				if (mDenoiseInputType == DenoiseInputType::L) {
					tFuncDenoise();
					tFunCalcError();
				} else if (mDenoiseInputType == DenoiseInputType::Error) {
					tFunCalcError();
					tFuncDenoise();
				}
			} else {
				tFunCalcError();
			}

			// [STEP#2] calculate the error sum

			float *sumPos  = network->getLossSumErrorGPUPtr();
			float *partSum = mNRRSParams.mTempGPUBuffer + (32 * 4); // warp size * 4
			calcSum2PassAsync<false>(errorPerPixel + resolutionSize, sumPos, partSum,
									 frameSizeScaledLength, stream);

			if (mDebugOn) {
				GPUCall(
					KRR_DEVICE_LAMBDA() { printf("[line 2294] error sum: %g\n", *sumPos); },
					stream);
			}

			if (sShowError) {

				// TODO: DEBUG ONLY
				// copy back error
				float *errorCPU = new float[resolutionSize];
				cudaMemcpy(errorCPU, errorPerPixel, sizeof(float) * resolutionSize,
						   cudaMemcpyDeviceToHost);

				// big => small
				std::sort(errorCPU, errorCPU + resolutionSize, std::greater<float>());
				const int k[] = {0, 10, 50, 100, 200, 500, 1000, 10000, 100000};
				for (int i = 0; i < std::size(k); i++) {
					Log(Info, "Error[%d]: %f", k[i], errorCPU[k[i]]);
				}
				for (int i = 0; i < std::size(k); i++) {
					Log(Info, "Error[%d]: %f", resolutionSize - 1 - k[i],
						errorCPU[resolutionSize - 1 - k[i]]);
				}

				float sum = 0, sum2 = 0;
				for (int i = 0; i < resolutionSize; i++) {
					sum += errorCPU[i];
				}
				float avg = sum / resolutionSize;
				Log(Info, "Error sum: %f", sum);
				Log(Info, "Error mean: %f", avg);
				for (int i = 0; i < resolutionSize; i++) {
					sum2 += abs(errorCPU[i] - avg) + abs(errorCPU[i]);
				}
				Log(Info, "Error variance: %f", sum2 / resolutionSize);

				float sumCPU = 0;
				cudaMemcpy(&sumCPU, sumPos, sizeof(float), cudaMemcpyDeviceToHost);
				Log(Info, "Error sum (GPU): %f", sumCPU);

				// error reduce sum
				float reduceSum = reduce_sum(errorPerPixel, resolutionSize, stream);
				Log(Info, "Error sum(tcnn::reduce_sum()): %f", reduceSum);

				delete[] errorCPU;

				// check how many zero L mean data
				uint *LMeanIsZero;
				cudaMalloc(&LMeanIsZero, 2 * sizeof(uint));
				cudaMemset(LMeanIsZero, 0, 2 * sizeof(uint));
				cudaDeviceSynchronize();
				GPUParallelFor(
					resolutionSize,
					KRR_DEVICE_LAMBDA(int itemIdx) {
						const uint pathStateLength = mPathState->size();
						while (itemIdx < pathStateLength) {
							NRRSRadianceRecordItem item = (*mPathState)[itemIdx];
							const RGB thpRGB = item.mThp.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
							const RGB LRGB	 = item.mL.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

							if (LRGB.mean() == 0) {
								atomicAdd(LMeanIsZero, 1);
							}
							RGB L = RGB::Zero();
							for (int ch = 0; ch < RGB::dim; ch++) {
								if (thpRGB[ch] > M_EPSILON) {
									L[ch] = LRGB[ch] / thpRGB[ch];
								}
							}

							if (L.mean() == 0) {
								atomicAdd(LMeanIsZero + 1, 1);
							}
							itemIdx += resolutionSize;
						}
					},
					stream);
				cudaDeviceSynchronize();
				GPUCall(
					KRR_DEVICE_LAMBDA() {
						float ratio	 = (*LMeanIsZero * 1.0f) / mPathState->size();
						float ratio2 = (*(LMeanIsZero + 1) * 1.0f) / mPathState->size();
						printf("LMeanIsZero[before /thp]: %d / %d = %f\n"
							   "LMeanIsZero[after  /thp]: %d / %d = %f\n",
							   *LMeanIsZero, mPathState->size(), ratio, *(LMeanIsZero + 1),
							   mPathState->size(), ratio2);
					},
					stream);
				cudaDeviceSynchronize();
				cudaFree(LMeanIsZero);
			}
		}

		const int debugPixel = mDebugTrainingData ? mDebugPixel : -1;

		NRRSNetworkOutputTraining1 *net1Output = network->getTrainingRRSNetOutputBufferPtr();

		// [STEP#3] generate training data
		if (mIgnoreSameTrainingDataForRRSNet) {
			cudaMemsetAsync(mPathStateNodeIdxAtomicBuffer, 0,
							sizeof(uint) * resolutionSize * mMaxRateForPathNodesBuffer, stream);
		}

		LinearKernel(nrrs_generate_training_data, stream, resolutionSize, mPathState, mTrainBuffer,
					 net1Output, mScene->getBoundingBox(), thp, errorForTraining,
					 errorForAvgForTraining, refMean, errorPerPixel,
					 // mRefImageDebug,
					 mRenderedImage, // TODO[IM]: use mRenderedImage instead of mRefImageDebug
					 mPixelState,
					 pixelIDArray, // here, save pixelId to sampleWeight
					 mNRRSParams.mIgnoreZeroL,
					 debugPixel, // [TODO] for debug
					 numSamples, mErrorImageScale, frameSizeWidth, mNumberSamplesThisPatch,
					 mCopyTheRRSNode, mPathStateNodeIdxAtomicBuffer,
					 mIgnoreSameTrainingDataForRRSNet);

		if (isStep3) {
			// [STEP#4] calculate the weight for each sample
			// invalid sample <= p2psSize
			uint p2psSize =
				*(uint *) (mNRRSParams.mTempCPUBuffer + mNRRSParams.mPathStateSizeInCPUBuffer);
			if (p2psSize == 0) {
				// sync as this is allocated by cudaMallocManaged()
				cudaStreamSynchronize(stream);
				p2psSize = mPathState->size();
				Log(Info, "Sync: p2psSize = 0, after Sync: %d", p2psSize);
			}

			// sampleWeight
			LinearKernel(nrrs_calculate_sample_weight, stream, p2psSize, sampleWeight, pixelIDArray,
						 mPixelState, mTrainBuffer);
		}
	}

	// cudaStreamSynchronize(stream); // [TODO] if error, uncomment this line
	sNumTrainingSamples	 = mTrainBuffer->size();
	const auto inputPos	 = mTrainBuffer->inputs();
	const auto outputPos = mTrainBuffer->outputs();

	uint numTrainBatches =
		min((uint) sNumTrainingSamples / mNRRSParams.mBatchSize + 1, mNRRSParams.mBatchPerFrame);

	float loss = 0.0f;
	network->updateReliefErrorScale();
	for (int iter = 0; iter < numTrainBatches; iter++) {
		size_t localBatchSize = min(sNumTrainingSamples - iter * mNRRSParams.mBatchSize,
									(size_t) mNRRSParams.mBatchSize);
		localBatchSize -= localBatchSize % 128;

		// [TODO] in fact, we should not drop any samples
		// drop the unaligned samples
		const int localBSPad = previous_multiple<int>(localBatchSize, BATCH_SIZE_GRANULARITY);
		if (localBSPad <= 0) {
			Log(Info, "Drop the small batch: %d", localBatchSize);
			continue;
		}

		const uint dataOffset = iter * mNRRSParams.mBatchSize;
		float *inputData	  = (float *) (inputPos + dataOffset);
		float *outputData	  = (float *) (outputPos + dataOffset);

		GPUMatrix_Float networkInputs(inputData, NRRS_LL2NET_DIM_INPUT, localBSPad);
		GPUMatrix_Float networkOutputs(outputData, NRRS_LL2NET_DIM_OUTPUT, localBSPad);
		{
			PROFILE("Train step");
			network->updateLossOffset(dataOffset);
			loss += network->train(networkInputs, networkOutputs, dataOffset, false);
		}
		if (mNRRSParams.mTrainOneBatch) {
			break;
		}
	}
	// sCurLossScalar.update(loss);
	sCurLossScalar.update(loss / numTrainBatches);
	sLossGraph[sNumLossSamples++ % LOSS_GRAPH_SIZE] = sCurLossScalar.emaVal();

	if (mExpOn && isnan(loss)) {
		gpContext->requestExit();
	}

	if (mAutoTrain && isnan(loss)) {
		mAutoTrainRequestExit = 2; // request exit
	}
}

void NRRSPathTracer::autoTrainControl() {
#define BB_CATCH(x) (timeConsumed > stateTime[x + 1])

	// auto train, now for grid search
	static int state	   = -1;
	static float startTime = (float) clock() / CLOCKS_PER_SEC;
	static float stateTime[5]{};

	float currentTime		 = (float) clock() / CLOCKS_PER_SEC;
	const float timeConsumed = currentTime - startTime;

	// if... if... but not if...else if... : to make sure the state time = 0 can be well handled
	if (state == -1) {
		// do once
		for (int i = 1; i < std::size(stateTime); ++i) {
			stateTime[i] = mAutoTrainStateTime[i - 1];
			stateTime[i] += stateTime[i - 1];
		}

		// startTime = (float) clock() / CLOCKS_PER_SEC;
		state	   = 0;
		mEnableRRS = false;
		mNRRSParams.mNet->setStep(0);
	}
	if (state == 0) {
		if (BB_CATCH(state)) {
			state = 1;

			mEnableRRS = false;
			mNRRSParams.mNet->setStep(1);
		}
	}
	if (state == 1) {
		if (BB_CATCH(state)) {
			state = 2;

			mEnableRRS							 = true;
			mNRRSParams.mLossUseNetworkOutputRRS = false;
			mNRRSParams.mNet->setStep(2);
		}
	}
	if (state == 2) {
		if (BB_CATCH(state)) {
			state = 3;

			mNRRSParams.mLossUseNetworkOutputRRS = true;
			mNRRSParams.mNet->setStep(3);
		}
	}

	if (state == 3) {
		// countdown for calculate time
		if (BB_CATCH(state)) {
			mNRRSParams.mStopTraining		   = true;
			constexpr int FRAMES_FOR_CALC_TIME = 100;
			static int sCountDown			   = FRAMES_FOR_CALC_TIME;
			static CpuTimer::TimePoint sStartTimeForCountDown{};
			if (sCountDown == FRAMES_FOR_CALC_TIME) {
				sStartTimeForCountDown = CpuTimer::getCurrentTimePoint();
			}
			if (--sCountDown < 0) {
				mAutoTrainTime1FramesInSeconds =
					CpuTimer::calcDuration(sStartTimeForCountDown,
										   CpuTimer::getCurrentTimePoint()) *
					1e-3 / FRAMES_FOR_CALC_TIME;
				state = 4;
			}
		}
	}

	if (state == 4) {
		// request exit
		mAutoTrainRequestExit = 1;
		state				  = 4;
	}

#undef BB_CATCH
}

void NRRSPathTracer::saveAndExit() {
	auto stream = gpContext->cudaStream;
	cudaStreamSynchronize(stream);

	auto resolutionSize = getFrameSize()[0] * getFrameSize()[1];

	// save json file
	json j;
	const float g1	   = mNRRSParams.mNet->gamma1();
	const float g2	   = mNRRSParams.mNet->gamma2();
	const float g3	   = mNRRSParams.mNet->gamma3();
	const float g4	   = mNRRSParams.mNet->gamma4();
	const float relMSE = mShowBuffer.mSumCPU * 1.0f / resolutionSize;
	const bool nan	   = mAutoTrainRequestExit != 1;

	j["gamma1"] = g1;
	j["gamma2"] = g2;
	j["gamma3"] = g3;
	j["gamma4"] = g4;
	j["relMSE"] = relMSE;
	j["nan"]	= nan;

	// get time stamp
	// filename: gamma1_gamma2_gamma3.json
	const auto filePath = "common/outputs/test/test/grid_search/";
	// if not exist, create the directory
	if (!fs::exists(filePath)) {
		fs::create_directories(filePath);
	}

	const float fps		   = 1.0f / mAutoTrainTime1FramesInSeconds;
	const float efficiency = relMSE * mAutoTrainTime1FramesInSeconds;

	std::ostringstream out;
	out << std::fixed << std::setprecision(4);
	out << filePath;
	out << int(nan) << "_" << efficiency << "_" << relMSE << "_" << fps << "_" << g1 << "_" << g2
		<< "_" << g3 << "_" << g4 << "_.json";
	std::ofstream o(out.str());
	o << std::setw(4) << j << std::endl;
	o.close();

	Log(Warning, "Save the json file (%s) and Exit", out.str().c_str());

	gpContext->requestExit();
}

// Use Best Strategy

void NRRSPathTracer::UBSRecordEfficiency(float efficiency) {
	// Log(Info, "state: %s, efficiency: %f", UBSState2String(mUBSLastSearchState).c_str(),
	// efficiency);

	if (efficiency > mUBSBestEfficiency) {
		mUBSBestEfficiency = efficiency;
		// attention, this is last search state
		mUBSBestState = mUBSLastSearchState;
	}
}

bool NRRSPathTracer::UBSNextState() {
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

void NRRSPathTracer::UBSAnalyseFrame() {
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

	LinearKernel(nrrs_UBS_calc_error, stream, resolutionSize, mUBSErrorBufferGPU,
				 // mRefImageDebug, // just for debug
				 mRenderedImage, // TODO[IM]: use mRenderedImage instead of mRefImageDebug
				 mPixelState, mSamplesPerPixel);

	// calculate error sum
	const float *rrsArray = mNRRSParams.mRRSArray;
	float *sum			  = mNRRSParams.mTempGPUBuffer;
	float *sumPos		  = sum + mNRRSParams.mUBSErrorSumPosInGPUBuffer;
	float *partSum		  = sum + (32 * 4); // warp size * 4

	calcSum2PassAsync<false>(mUBSErrorBufferGPU, sumPos, partSum, resolutionSize, stream);
	cudaMemcpyAsync(&mUBSLastError, sumPos, sizeof(float), cudaMemcpyDeviceToHost, stream);
	// cudaMemcpy(&mUBSLastError, sumPos, sizeof(float), cudaMemcpyDeviceToHost);

	return;
}

// template declaration
template void NRRSPathTracer::generateRRSNumber<true, false>(const int depth);
template void NRRSPathTracer::generateRRSNumber<false, true>(const int depth);
template void NRRSPathTracer::generateRRSNumber<false, false>(const int depth);
template void NRRSPathTracer::writeResultToRenderTarget<true>(RenderContext *context);
template void NRRSPathTracer::writeResultToRenderTarget<false>(RenderContext *context);

KRR_REGISTER_PASS_DEF(NRRSPathTracer);
NAMESPACE_END(krr)