#pragma once
#include <atomic>

#include "common.h"
#include "device/cuda.h"
#include "logger.h"
#include "util/math_utils.h"

#include "workitem.h"
#include "render/profiler/profiler.h"
#include "render/simplewpt/workqueue.h"

NAMESPACE_BEGIN(krr)

struct NRCPathPixelStateBuffer : public SOA<NRCPathPixelState> {
public:
	NRCPathPixelStateBuffer() = default;
	NRCPathPixelStateBuffer(int n, Allocator alloc) : SOA<NRCPathPixelState>(n, alloc) {}

	KRR_CALLABLE void reset(int pixelId) {
		// reset a guided state (call when begining a new frame...
		mCurDepth[pixelId] = 0;
	}

	/* Records raw (unnormalized) vertex data along the path of the current pixel */
	KRR_CALLABLE void
	incrementDepth(int pixelId,
				   const Ray &ray,		// current scattered ray
				   Spectrum &thp,		// current throughput
				   Spectrum &LSpectrum, // current radiance
				   bool delta = false,	// is this scatter event sampled from a delta lobe?
				   /* below optional / auxiliary data */
				   const SurfaceInteraction &intr = {} // may obtain other auxiliary data from this
	) {
		int depth = mCurDepth[pixelId];
		if (depth >= NRC_MAX_TRAIN_DEPTH) {
			return;
		}
		const RGB Lrgb					= LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
		const RGB thpRGB				= thp.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
		mRecords[depth].mL[pixelId]		= Lrgb;
		mRecords[depth].mThp[pixelId]	= thp;
		mRecords[depth].mPos[pixelId]	= ray.origin;
		mRecords[depth].mDir[pixelId]	= utils::worldToLatLong(ray.dir);
		mRecords[depth].mDelta[pixelId] = delta;
		mCurDepth[pixelId]				= depth + 1;
#if NETWORK_AUXILIARY_INPUT
		mRecords[depth].mNormal[pixelId]	= utils::worldToLatLong(intr.n);
		mRecords[depth].mRoughness[pixelId] = intr.sd.roughness;
#endif
	}

	/* Two types of radiance contribution call this routine:
		Emissive intersection and Next event estimation. */
	KRR_CALLABLE void recordRadiance(int pixelId, Spectrum &LSpectrum) {
		int depth	   = min(mCurDepth[pixelId], (uint) NRC_MAX_TRAIN_DEPTH);
		const RGB Lrgb = LSpectrum.toRGB({}, *KRR_DEFAULT_COLORSPACE_GPU);
		for (int i = 0; i < depth; i++) {
			// local radiance should be obtained via L / thp.
			const RGB &prev			= mRecords[i].mL[pixelId];
			mRecords[i].mL[pixelId] = prev + Lrgb;
		}
	}
};

class NRCHitLightRayQueue : public WorkQueue<NRCHitLightWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class NRCScatterRayQueue : public WorkQueue<NRCScatterRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class NRCRayQueue : public WorkQueue<NRCRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue; // use parent constructor

	KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
		int index			  = allocateEntry();
		this->mA0[index]	  = 1;
		this->mAn[index]	  = 0;
		this->mDepth[index]	  = 0;
		this->mThp[index]	  = Spectrum::Ones();
		this->mPixelId[index] = pixelId;
		this->mRay[index]	  = ray;
		return index;
	}
};

class NRCMissRayQueue : public WorkQueue<SWPTMissRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(NRCRayWorkItem w) {
		return push(
			SWPTMissRayWorkItem{w.mRay, w.mCtx, w.mPdf, w.mThp, w.mBsdfType, w.mDepth, w.mPixelId});
	}
};

class NRCInferenceQueue : public WorkQueue<NRCInferenceWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

NAMESPACE_END(krr)