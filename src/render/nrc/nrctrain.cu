#include "nrcparameters.h"
#include "nrctrain.h"
#include "util/math_utils.h"

NAMESPACE_BEGIN(krr)

/* Input data layout:
 *	| spatial input (3) | directional input (2) | [opt] auxiliary input (3) |
 *  |					                        | for learning auxiliary    |
 */

__global__ void
nrc_generate_training_data(const size_t nElements, uint trainPixelOffset, uint trainPixelStride,
						   NetworkTrainBuffer<NRCNetworkInput, NRCNetworkOutput> *trainBuffer,
						   NRCPathPixelStateBuffer *guidedState, const AABB sceneAABB) {
	// this costs about 0.5ms
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) {
		return;
	}

	int pixelId = trainPixelOffset + tid * trainPixelStride;

	int depth = guidedState->mCurDepth[pixelId];
	for (int curDepth = 0; curDepth < depth; curDepth++) {
		NRCNetworkInput input	= {};
		NRCNetworkOutput output = {};

		const NRCRadianceRecordItem &record = guidedState->mRecords[curDepth][pixelId];
		if (record.mDelta) {
			continue; // do not incorporate samples that from a delta lobe.
		}
		input.mPos = normalizeSpatialCoord(record.mPos, sceneAABB);
		input.mDir = record.mDir;

#if NETWORK_AUXILIARY_INPUT > 0
		input.mAuxiliary[0] = record.mNormal[0];
		input.mAuxiliary[1] = record.mNormal[1];
		input.mAuxiliary[2] = nrc_warp_roughness_for_ob(record.mRoughness);
#endif
		RGB L = RGB::Zero();
		for (int ch = 0; ch < RGB::dim; ch++) {
			if (record.mThp[ch] > M_EPSILON) {
				L[ch] = record.mL[ch] / record.mThp[ch];
			}
		}
		output.mL = L;
		if (!(input.mPos.hasNaN() || input.mDir.hasNaN() || L.hasNaN())) {
			trainBuffer->push(input, output);
		}
		// else printf("Find invalid training sample! (quite not expected...\n");
	}
}

__global__ void nrc_generate_inference_data(const size_t nElements, NRCInferenceQueue *inferQueue,
											float *data, const AABB sceneAABB) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= inferQueue->size()) {
		return;
	}
	uint data_idx					 = i * NRC_DIM_INPUT;
	const NRCInferenceWorkItem &item = inferQueue->operator[](i).operator NRCInferenceWorkItem();
	Vector3f pos					 = normalizeSpatialCoord(item.mPos, sceneAABB);
	*(Vector3f *) &data[data_idx]	 = pos;
	*(Vector2f *) &data[data_idx + NRC_DIM_SPATIAL_INPUT] = worldToLatLong(item.mDir);
#if NETWORK_AUXILIARY_INPUT
	*(Vector2f *) &data[data_idx + NRC_DIM_SPATIAL_INPUT + NRC_DIM_DIRECTIONAL_INPUT] =
		item.mNormal;
	data[data_idx + NRC_DIM_SPATIAL_INPUT + NRC_DIM_DIRECTIONAL_INPUT + 2] =
		nrc_warp_roughness_for_ob(item.mRoughness);
#endif
}

NAMESPACE_END(krr)