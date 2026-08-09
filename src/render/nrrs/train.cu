#include "train.h"
#include "render/ears/workitem.h"
#include "render/ears/workqueue.h"
#include "net/networkcommon.h"

NAMESPACE_BEGIN(krr)
__global__ void check_nan(const size_t nElements, const float *data) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) {
		return;
	}

	if (isnan(data[tid])) {
		printf("nan detected at %llu\n", tid);
	}
}

// make sure abs(input) is not too large
static __device__ float box_cox_transform(const float x) {
	constexpr float lambda = 0.5f;
	if (lambda == 0.0f) {
		return logf(x);
	} else if (lambda == 0.5f) {
		return 2.0f * (sqrt(x) - 1.0f);
	} else {
		return (powf(x, lambda) - 1.0f) / lambda;
	}
}

// make sure the input is positive
static __device__ RGB box_cox_transform(const RGB &x) {
	return RGB(box_cox_transform(x[0]), box_cox_transform(x[1]), box_cox_transform(x[2]));
}

static __device__ void box_cox_transform(__half *target, const RGB &source) {
	target[0] = box_cox_transform(source[0]);
	target[1] = box_cox_transform(source[1]);
	target[2] = box_cox_transform(source[2]);
}

__global__ void
nrrs_generate_inference_data_from_0to1(const size_t nElements, const NRRSInferenceQueue *inferQueue,
									   const float *net0Input, const __half *prediction_l,
									   const __half *prediction_l2, MyFilm *renderedImage,
									   const uint *pixelID, NRRSNetworkInput1 *net1Input) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	// nElements is upbound, we should use inferQueue->size() as the upbound
	if (tid >= inferQueue->size()) {
		return;
	}

	RGB thp = inferQueue->mThp[tid];

#ifdef NRRS_USE_REF_MEAN
	uint stid	  = inferQueue->mScatterQueueIndex[tid];
	uint pixelId  = pixelID[stid];
	float refMean = renderedImage->getPixel(pixelId).head<3>().mean();
#endif

	// here is also 16 as we use inference_mixed_precision()
	// if we use inference(), then it should be NRRS_LL2NET_DIM_OUTPUT
	const __half *pred_l  = prediction_l + tid * NRRS_LL2NET_DIM_OUTPUT_PADDED;
	const __half *pred_l2 = prediction_l2 + tid * NRRS_LL2NET_DIM_OUTPUT_PADDED;

	const NRRSNetworkInput0 *dataBase = ((NRRSNetworkInput0 *) net0Input) + tid;

#ifdef NRRS_USE_POS_DIR
	Vector3f pos = dataBase->mPos;
	Vector2f dir = dataBase->mDir;
#endif

	RGB l  = max(RGB(pred_l[0], pred_l[1], pred_l[2]), RGB(0));
	RGB l2 = max(RGB(pred_l2[0], pred_l2[1], pred_l2[2]), RGB(0));

	// constexpr float maxRGB = 100.0f;
	// l					   = min(l, RGB(maxRGB));
	// l2					   = min(l2, RGB(maxRGB));

	NRRSNetworkInput1 *input = net1Input + tid;
	box_cox_transform(input->mL, l);
	box_cox_transform(input->mL2, l2);
	box_cox_transform(input->mThp, thp);
#ifdef NRRS_USE_ROUGHNESS
	input->mRoughness = dataBase->mRoughness;
#endif

#ifdef NRRS_USE_REF_MEAN
	input->mRefMean = box_cox_transform(refMean);
#endif
#ifdef NRRS_USE_POS_DIR
	input->mPos = pos;
	input->mDir = dir;
#endif
	for (int i = 0; i < NRRS_RRSNET_DIM_INPUT_PADDING; i++) {
		input->mPadding0[i] = i == 0 ? 1.0f : 0.0f;
	}

	// throughput x Li
	// throughput / refMean
	// Li / refMean
	// throughput x Li / refMean
	// const float refMeanInv = 1.0f / (refMean + 1e-2f);
	// const float IpxEst	   = (thp * l).mean();
	// input->mPadding0[0]	   = IpxEst;
	// input->mPadding0[1]	   = thp.mean() * refMeanInv;
	// input->mPadding0[2]	   = l.mean() * refMeanInv;
	// input->mPadding0[3]	   = IpxEst * refMeanInv;
	// for (int i = 0; i < 4; i++) {
	//	input->mPadding0[i] = box_cox_transform(input->mPadding0[i]);
	//}
	// input->mPadding0[4] = 1.0f;
}

__global__ void nrrs_generate_training_data_from_0to1_aid(
	const size_t nElements, NRRSNetworkInput1AID *net1Input, const float *net0Input,
	const __half *predictionL, const __half *predictionL2, const float *refMeanForTraining,
	const float *thpForTraining, __half *ll2PtrForTraining) {

	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nElements) {
		return;
	}

	// [STEP#1] make training data(L/L2)
	// padding to 16, see BB_NETWORK_CHECK_PADDING in network.cpp
	constexpr uint32_t net0OutputWidth = 16;
	const __half *pred_l			   = predictionL + tid * net0OutputWidth;
	const __half *pred_l2			   = predictionL2 + tid * net0OutputWidth;

	__half *ll2Ptr = ll2PtrForTraining + tid * 6;
	for (int tmpi = 0; tmpi < 3; tmpi++) {
		ll2Ptr[tmpi]	 = pred_l[tmpi];
		ll2Ptr[tmpi + 3] = pred_l2[tmpi];
	}

	// [STEP#2] make input
	const NRRSNetworkInput0 *dataBase = ((NRRSNetworkInput0 *) net0Input) + tid;

	NRRSNetworkInput1AID *input = net1Input + tid;
	input->mPos					= dataBase->mPos;
	input->mDir					= dataBase->mDir;

#ifdef NRRS_USE_ROUGHNESS
	input->mRoughness = dataBase->mRoughness;
#endif
#ifdef NRRS_USE_REF_MEAN
	float refMean	= refMeanForTraining[tid];
	input->mRefMean = box_cox_transform(refMean);
#endif
	RGB *thp	= (RGB *) (thpForTraining + tid * RGB::dim);
	input->mThp = box_cox_transform(*thp);
}

__global__ void nrrs_generate_training_data_from_0to1(
	const size_t nElements, NRRSNetworkInput1 *net1Input, const float *net0Input,
	const __half *predictionL, const __half *predictionL2, const float *refMeanForTraining,
	const float *thpForTraining, __half *ll2PtrForTraining, const bool trainSigma) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nElements) {
		return;
	}

	// padding to 16, see BB_NETWORK_CHECK_PADDING in network.cpp
	constexpr uint32_t net0OutputWidth = 16;

	const float *dataBase = net0Input + tid * NRRS_LL2NET_DIM_INPUT;
#ifdef NRRS_USE_POS_DIR
	const Vector3f *pos = (Vector3f *) (dataBase + NRRS_LL2NET_INPUT_POS_OFFSET);
	const Vector2f *dir = (Vector2f *) (dataBase + NRRS_LL2NET_INPUT_DIR_OFFSET);
#endif

#ifdef NRRS_USE_ROUGHNESS
	const float *roughness = dataBase + NRRS_LL2NET_INPUT_ROUGHNESS_OFFSET;
#endif

	RGB *thp = (RGB *) (thpForTraining + tid * RGB::dim);

#ifdef NRRS_USE_REF_MEAN
	float refMean = refMeanForTraining[tid];
#endif

	const __half *pred_l  = predictionL + tid * net0OutputWidth;
	const __half *pred_l2 = predictionL2 + tid * net0OutputWidth;

	__half *ll2Ptr = ll2PtrForTraining + tid * 6;
	for (int tmpi = 0; tmpi < 3; tmpi++) {
		ll2Ptr[tmpi]	 = pred_l[tmpi];
		ll2Ptr[tmpi + 3] = pred_l2[tmpi];
	}

	RGB l  = max(RGB(*pred_l, *(pred_l + 1), *(pred_l + 2)), RGB(0));
	RGB l2 = max(RGB(*pred_l2, *(pred_l2 + 1), *(pred_l2 + 2)), RGB(0));

	if (trainSigma) {
		l2[0] = activationSigma(l2[0]);
		l2[1] = activationSigma(l2[1]);
		l2[2] = activationSigma(l2[2]);
	}

	// constexpr float maxRGB = 100.0f;
	// l					   = min(l, RGB(maxRGB));
	// l2					   = min(l2, RGB(maxRGB));

	NRRSNetworkInput1 *input = net1Input + tid;
	box_cox_transform(input->mL, l);
	box_cox_transform(input->mL2, l2);
	box_cox_transform(input->mThp, *thp);

#ifdef NRRS_USE_ROUGHNESS
	input->mRoughness = *roughness;
#endif
#ifdef NRRS_USE_REF_MEAN
	input->mRefMean = box_cox_transform(refMean);
#endif
#ifdef NRRS_USE_POS_DIR
	input->mPos = *pos;
	input->mDir = *dir;
#endif

	for (int i = 0; i < NRRS_RRSNET_DIM_INPUT_PADDING; i++) {
		input->mPadding0[i] = i == 0 ? 1.0f : 0.0f;
	}

	// throughput x Li
	// throughput / refMean
	// Li / refMean
	// throughput x Li / refMean
	// const float refMeanInv = 1.0f / (refMean + 1e-2f);
	// const float IpxEst	   = (*thp * l).mean();
	// input->mPadding0[0]	   = IpxEst;
	// input->mPadding0[1]	   = (*thp).mean() * refMeanInv;
	// input->mPadding0[2]	   = l.mean() * refMeanInv;
	// input->mPadding0[3]	   = IpxEst * refMeanInv;
	// for (int i = 0; i < 4; i++) {
	//	input->mPadding0[i] = box_cox_transform(input->mPadding0[i]);
	//}
	// input->mPadding0[4] = 1.0f;
}

__global__ void nrrs_generate_inference_data_aid(const size_t nElements,
												 const NRRSInferenceQueue *inferQueue,
												 NRRSNetworkInput1AID *net1Input,
												 const AABB sceneAABB, MyFilm *renderedImage,
												 const uint *pixelID) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= inferQueue->size()) {
		return;
	}

	const NRRSInferenceWorkItem item = (*inferQueue)[tid];

	NRRSNetworkInput1AID *input = net1Input + tid;

	input->mPos = normalizeSpatialCoord(item.mPos, sceneAABB);
	input->mDir = item.mDir;
#ifdef NRRS_USE_ROUGHNESS
	input->mRoughness = warp_roughness(item.mRoughness);
#endif
#ifdef NRRS_USE_REF_MEAN
	uint pixelId	= pixelID[item.mScatterQueueIndex];
	float refMean	= renderedImage->getPixel(pixelId).head<3>().mean();
	input->mRefMean = box_cox_transform(refMean);
#endif
	input->mThp = box_cox_transform(item.mThp);
}

__global__ void nrrs_generate_inference_data(const size_t nElements,
											 const NRRSInferenceQueue *inferQueue, float *net0Input,
											 const AABB sceneAABB) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= inferQueue->size()) {
		return;
	}

	uint data_idx					 = tid * NRRS_LL2NET_DIM_INPUT;
	const NRRSInferenceWorkItem item = (*inferQueue)[tid];

	Vector3f pos			 = normalizeSpatialCoord(item.mPos, sceneAABB);
	NRRSNetworkInput0 *input = (NRRSNetworkInput0 *) (net0Input + data_idx);
	input->mPos				 = pos;
	input->mDir				 = item.mDir;
#ifdef NRRS_USE_ROUGHNESS
	input->mRoughness = warp_roughness(item.mRoughness);
#endif
}

__global__ void upload_denoise_error_buffer(const size_t nElements, const bool chooseError,
											const float *errorPerPixel, const RGB *weightedL,
											RGB *color) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) {
		return;
	}

	if (chooseError) {
		const float error = errorPerPixel[tid];
		color[tid]		  = RGB(error);
	} else {
		color[tid] = weightedL[tid];
	}
}

__global__ void download_denoise_error_buffer(const size_t nElements, const bool chooseError,
											  float *errorPerPixel, RGB *weightedL,
											  const RGB *color, const bool errorMultiplySamples,
											  float *numberSamplesThisPatch,
											  const uint resolutionSize, float *errorFactorXX,
											  float *errorFactorXY) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) {
		return;
	}
	if (chooseError) {
		float error		   = color[tid].mean();
		errorPerPixel[tid] = error;

		if (errorMultiplySamples) {
			const float spp = numberSamplesThisPatch[tid];
			// spp >= 1 as CameraRay add 1
			// error = error / spp;
			error = error * sqrtf((float) spp); // [!!2pos!!]

			// float x	  = 1.0f / spp;
			// float txy = errorFactorXY[tid] + error * x;
			// float txx = errorFactorXX[tid] + x * x;
			// float k	  = txy / txx;

			// errorFactorXX[tid] += txx;
			// errorFactorXY[tid] += txy;

			// error = error / sqrtf(k); // [!!2pos!!]

			// numberSamplesThisPatch[tid] = k;
		}
		errorPerPixel[tid + resolutionSize] = error;

	} else {
		weightedL[tid] = color[tid];
	}
}

__global__ void nrrs_update_weighted_L(const size_t nElements, NRRSPixelStateBuffer *psBuffer,
									   const RGB *weightedLCurrent, const uint *weightedLCurrentAcc,
									   RGB *weightedL, const float blendWeightForError) {

	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) {
		return;
	}

	int spp = weightedLCurrentAcc[tid];
	if (spp == 0) {
		printf("[ERROR] nrrs_update_weighted_L(): spp is zero\n");
	}
	RGB LCurrent = weightedLCurrent[tid] / spp;

	RGB LBlend = weightedL[tid];

	RGB LResult	   = LBlend * (1.0f - blendWeightForError) + LCurrent * blendWeightForError;
	weightedL[tid] = LResult;
}

__global__ void nrrs_calculate_error(const size_t nElements, float *error, MyFilm *renderedImage,
									 RGB *renderedImageDenoised, const bool useAccBuffer,
									 NRRSPixelStateBuffer *psBuffer, const RGB *weightedL,
									 const bool clamp, const bool errorMultiplySamples,
									 float *numberSamplesThisPatch, const uint resolutionSize,
									 float *errorFactorXX, float *errorFactorXY,
									 const bool denoisedOn) {

	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) {
		return;
	}

	// here do not perform transform
	RGB ref;
	if (useAccBuffer) {
		ref = renderedImage->getPixelNoTransform(tid).head<3>();
	} else {
		ref = renderedImageDenoised[tid];
	}
	RGB L = weightedL[tid];

	// relative error, avoid emphasizing the error in the dark region
	// float err = ((L - ref) / (ref + 1e-2f)).square().mean();
	float err = ((L - ref).square() / (ref.square() + 1e-2f)).mean();

	// absolute error
	// float err = (L - ref).square().mean();

	// float err = psBuffer->mError[tid] / psBuffer->mAccSppForError;

	// err = logf(err + 1.0f);

	if (clamp) {
		// clamp < 100.0f
		// err = min(err, 1.0f);
		// err = max(err, 0.01f);
		err = min(err, 100.0f);
	}

	// err = log(err + 1.0f);

	error[tid] = err;

	if (errorMultiplySamples) {
		const float spp = numberSamplesThisPatch[tid];
		err				= err * sqrtf(spp); // [!!2pos!!]

		// float xx = errorFactorXX[tid];
		// float xy = errorFactorXX[tid];
		//// spp >= 1 as CameraRay add 1
		//// err = err / spp;

		// float k = 1.0f;
		// if (xx > 0) {
		//	k = sqrtf(xy / xx);
		// }

		// err = err / sqrtf(k); // [!!2pos!!]

		// if (!denoisedOn) {
		//	numberSamplesThisPatch[tid] = k;
		// }
	}
	error[tid + resolutionSize] = err;
}

__global__ void nrrs_generate_training_data(
	const size_t nElements, NRRSPathNodesBuffer *pathState,
	NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0> *trainBuffer,
	NRRSNetworkOutputTraining1 *net1Output, const AABB sceneAABB, float *thpForTraining,
	float *errorForTraining, float *errorForAvgForTraining, float *refMeanForTraining,
	float *errorPixel, MyFilm *renderedImage, NRRSPixelStateBuffer *pixelState, uint *rayPixelID,
	const bool ignoreZeroL, const int debugPixelID, float *numSamples, const uint errorImageScale,
	const uint frameSizeWidth, const float *numberSamplesThisPatch, const bool copyTheRRSNode,
	uint *pathStateNodeIdxAtomicBuffer, const bool ignoreSameTrainingDataForRRSNet) {

	uint itemIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (itemIdx >= nElements) {
		return;
	}

	const uint resolutionSize = nElements;

	const uint pathStateLength = pathState->size();

	const bool shouldDebug = debugPixelID != -1;

	// note here we use tid
	// Sampler sampler = &pixelState->mSampler[itemIdx];

	while (itemIdx < pathStateLength) {
		NRRSNetworkInput0 input			  = {};
		NRRSNetworkOutputTraining0 output = {};

		NRRSRadianceRecordItem item = pathState->operator[](itemIdx);

		bool sameNodeForRRSNet = false;
		if (ignoreSameTrainingDataForRRSNet && item.mLastNodeIdx != -1) {
			sameNodeForRRSNet =
				atomicAdd(pathStateNodeIdxAtomicBuffer + item.mLastNodeIdx, 1u) != 0;
		}

		RGB thpRGB	   = item.mThp.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
		const RGB LRGB = item.mL.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

		input.mDir = item.mDir;
		input.mPos = normalizeSpatialCoord(item.mPos, sceneAABB);

#ifdef NRRS_USE_ROUGHNESS
		input.mRoughness = warp_roughness(item.mRoughness);
#endif

		RGB L = RGB::Zero();
		for (int ch = 0; ch < RGB::dim; ch++) {
			if (thpRGB[ch] > M_EPSILON) {
				L[ch] = LRGB[ch] / thpRGB[ch];
			}
		}
		L = max(L, RGB::Zero());

		// TODO: you should add a error for each sample(put an array in loss)
		// output.error = errorPixel[item.mPixelId];

		// RGB jitter1 = RGB(sampler.get1D(), sampler.get1D(), sampler.get1D()) * 2.0f - 1.0f;
		// RGB jitter2 = RGB(sampler.get1D(), sampler.get1D(), sampler.get1D()) * 2.0f - 1.0f;

		output.mL  = L;
		output.mL2 = L * L; // in fact, not used

		// ignore deep samples
		// if (item.mDepth >= 4) {
		//	output.mL  = -1.0f * RGB::Ones();
		//	output.mL2 = -1.0f * RGB::Ones();
		// }

		const float rrs = item.mRRS;

		if (copyTheRRSNode) {
			// [IM] here we use thp before rrs performed
			thpRGB *= rrs;
		}

		bool keepData = !ignoreZeroL || (L.mean() > 0);
		if (keepData) {
			if (!(input.mPos.hasNaN() || input.mDir.hasNaN() || L.hasNaN())) {
				// RGB refI = RGB(renderedImage->getPixel(item.mPixelId));
				//  make sure the element of refI is not zero
				//  refI[0]				= max(refI[0], 1e-2f);
				//  refI[1]				= max(refI[1], 1e-2f);
				//  refI[2]				= max(refI[2], 1e-2f);
				// const RGB thpRGBrel = thpRGB / refI;
				// input.mThp			   = thpRGBrel; // use relative thp

				auto idxInTrainBuffer = trainBuffer->push(input, output);
				if (sameNodeForRRSNet) {
					net1Output[idxInTrainBuffer].mRRS = -1.0f;
					itemIdx += resolutionSize;
					continue;
				}

				RGB *thpForTrainingPtr = (RGB *) (thpForTraining + idxInTrainBuffer * RGB::dim);
				*thpForTrainingPtr	   = thpRGB;
				//*thpForTrainingPtr	   = thpRGBrel;
				refMeanForTraining[idxInTrainBuffer] =
					renderedImage->getPixel(item.mPixelId).head<3>().mean();
				rayPixelID[idxInTrainBuffer] = item.mPixelId;
				uint *ps					 = pixelState->mTrainingSamples;
				atomicAdd(&ps[item.mPixelId], 1);

				net1Output[idxInTrainBuffer].mRRS = rrs;

				// special case for error
				int pixelIdScaled =
					offsetFrame2Scaled(item.mPixelId, frameSizeWidth, errorImageScale);
				errorForTraining[idxInTrainBuffer] = errorPixel[pixelIdScaled];
				errorForAvgForTraining[idxInTrainBuffer] =
					errorPixel[pixelIdScaled + resolutionSize];
				numSamples[idxInTrainBuffer] = numberSamplesThisPatch[pixelIdScaled];

				if (shouldDebug && (debugPixelID == item.mPixelId)) {
					printf("itemIdx: %d\n"
						   "debugPixelID: %d\n"
						   "item.mPixelId: %u\n"
						   "errorForTraining[idxInTrainBuffer]: %g\n"
						   "refMeanForTraining[idxInTrainBuffer]: %g\n"
						   "rayPixelID[idxInTrainBuffer]: %d\n"
						   "numSamples[idxInTrainBuffer]: %g\n"
						   "ps[item.mPixelId]: %d\n"
						   "errorForTraining/numSamples: %g\n",
						   itemIdx, debugPixelID, item.mPixelId, errorForTraining[idxInTrainBuffer],
						   refMeanForTraining[idxInTrainBuffer], rayPixelID[idxInTrainBuffer],
						   numSamples[idxInTrainBuffer], ps[item.mPixelId],
						   errorForTraining[idxInTrainBuffer] / numSamples[idxInTrainBuffer]);
				}

			} else {
				printf("[nrrs_generate_training_data] meet nan, pos = (%f, %f, %f), "
					   "dir = (%f, %f), L = (%f, %f, %f)\n",
					   input.mPos.x(), input.mPos.y(), input.mPos.z(), input.mDir.x(),
					   input.mDir.y(), L[0], L[1], L[2]);
			}
		}
		itemIdx += resolutionSize;
	}
}

__global__ void nrrs_calculate_sample_weight(
	const size_t nElements, float *sampleWeight, const uint *pixelIDArray,
	NRRSPixelStateBuffer *psBuffer,
	NetworkTrainBuffer<NRRSNetworkInput0, NRRSNetworkOutputTraining0> *trainBuffer) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	// more tight
	if (tid >= trainBuffer->size()) {
		return;
	}

	const uint pixelId = pixelIDArray[tid];

	const uint rays	  = psBuffer->mTrainingSamples[pixelId];
	sampleWeight[tid] = rays ? 1.0f / rays : 0;

	// here we do not have to check if the number of rays is zero
	// if (rays == 0) {
	// printf("[ERROR] nrrs_calculate_sample_weight(): rays is zero\n");
	//}
}

__global__ void nrrs_UBS_calc_error(const size_t nElements, float *error, MyFilm *renderedImage,
									NRRSPixelStateBuffer *psBuffer, const int spp) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) {
		return;
	}

	RGB ref			   = renderedImage->getPixel(tid).head<3>();
	Spectrum LSpectrum = psBuffer->mL[tid] / spp;
	RGB L			   = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);

	// relative error, avoid emphasizing the error in the dark region
	float err  = ((L - ref).square() / (ref.square() + 1e-2f)).mean();
	error[tid] = err;
}

__global__ void calculate_gradient_sum(const uint32_t n_elements, __half *__restrict__ gradient_1,
									   const __half *__restrict__ gradient_2) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_elements) return;
	gradient_1[tid] += gradient_2[tid];
}

__global__ void gen_for_l2_evauate(const uint32_t n_elements, __half *__restrict__ nelLOutput,
								   float *__restrict__ target) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_elements) return;
	// TODO
}

// template initialization

NAMESPACE_END(krr)
