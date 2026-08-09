#pragma once
#include "common.h"
#include <atomic>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "device/atomic.h"
#include "logger.h"
#include "workitem.h"
#include "render/wavefront/workqueuebasic.h"

NAMESPACE_BEGIN(krr)

class SWPTPixelStateBuffer : public SOA<SWPTPixelState> {
public:
	SWPTPixelStateBuffer() = default;
	SWPTPixelStateBuffer(int n, Allocator alloc) : SOA<SWPTPixelState>(n, alloc) {}

	KRR_CALLABLE void setRadiance(int pixelId, Spectrum L_val) { mL[pixelId] = L_val; }
	KRR_CALLABLE void addRadiance(int pixelId, Spectrum L_val) {
		L_val		= L_val + Spectrum(mL[pixelId]);
		mL[pixelId] = L_val;
	}
	KRR_DEVICE void addRadianceAtomic(int pixelId, Spectrum L_val) {
#ifdef KRR_DEVICE_CODE
		atomicAdd(((float *) &mL[pixelId]) + 0, L_val.x());
		atomicAdd(((float *) &mL[pixelId]) + 1, L_val.y());
		atomicAdd(((float *) &mL[pixelId]) + 2, L_val.z());
#else
		Log(Fatal, "Not Implemented");
#endif
	}
};

class SWPTRayQueue : public WorkQueue<SWPTRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue; // use parent constructor

	KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
		int index			  = allocateEntry();
		this->mDepth[index]	  = 0;
		this->mThp[index]	  = Spectrum::Ones();
		this->mPixelId[index] = pixelId;
		this->mRay[index]	  = ray;
		return index;
	}
};

class SWPTMissRayQueue : public WorkQueue<SWPTMissRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class SWPTHitLightRayQueue : public WorkQueue<SWPTHitLightWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class SWPTShadowRayQueue : public WorkQueue<SWPTShadowRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class SWPTScatterRayQueue : public WorkQueue<SWPTScatterRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	inline uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int offset = offsetof(SWPTScatterRayQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((char *) &(this->m_size)) - ((char *) this));
			Log(Info, "[SWPTScatterRayQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

NAMESPACE_END(krr)