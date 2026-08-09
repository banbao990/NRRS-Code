#pragma once
#include <atomic>

#include "common.h"
#include "device/cuda.h"
#include "logger.h"
#include "util/math_utils.h"

#include "render/profiler/profiler.h"
#include "workitem.h"
#include "render/wavefront/workqueue.h"

NAMESPACE_BEGIN(krr)

class EARSRayQueue : public WorkQueue<EARSRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue; // use parent constructor

	KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
		int index = allocateEntry();

		mDepth[index]	= 0;
		mNodeIdx[index] = -1;
		mThp[index]		= Spectrum::Ones();
		mPixelId[index] = pixelId;
		mRay[index]		= ray;

		return index;
	}

	// KRR_CALLABLE int allocAll(uint size) {
	//	int index = m_size.fetch_add(size);
	//	return index;
	// }
};

class EARSMissRayQueue : public WorkQueue<EARSMissRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class EARSHitLightRayQueue : public WorkQueue<EARSHitLightWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class EARSShadowRayQueue : public WorkQueue<EARSShadowRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class EARSScatterRayQueue : public WorkQueue<EARSScatterRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	inline uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(TidQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((char *) &(m_size)) - ((char *) this));
			Log(Info, "[EARSScatterRayQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

class EARSInferenceQueue : public WorkQueue<EARSInferenceItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	inline uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(TidQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((char *) &(m_size)) - ((char *) this));
			Log(Info, "[EARSInferenceQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

class EARSPathNodesBuffer : public SOA<EARSRadianceRecordItem> {
public:
	EARSPathNodesBuffer() = default;
	EARSPathNodesBuffer(int n, Allocator alloc) : SOA<EARSRadianceRecordItem>(n, alloc) {}

	KRR_DEVICE void reset(const bool log) {
		if (log && mInvalidNode > 0) {
			printf("[Nodes] Last Statistic: mIndex = %d, mInvalidNode = %d\n", mIndex,
				   mInvalidNode);
		}
		mIndex		 = 0;
		mInvalidNode = 0;
	}

	KRR_DEVICE void recordRadiance(uint nodeIdx, const Spectrum &L, const bool isNeeRay) {
		static_assert(Spectrum::dim == 3, "only support 3 channels");
		if (nodeIdx >= mIndex || nodeIdx == -1) {
			return;
		}
		uint depth = 1;

		while (nodeIdx != -1) {
			// record the radiance
			atomicAdd(((float *) &(mL[nodeIdx])) + 0, L.x());
			atomicAdd(((float *) &(mL[nodeIdx])) + 1, L.y());
			atomicAdd(((float *) &(mL[nodeIdx])) + 2, L.z());

			// uint cost = depth;
			uint cost = isNeeRay ? 1u : depth;

			atomicAdd(((uint *) &(mCost[nodeIdx])), cost);
			// atomicAdd(((uint *) &(mNumSamples[nodeIdx])), 1u);
			++depth;
			nodeIdx = mLastNodeIdx[nodeIdx];
		}
	}

	KRR_DEVICE int recordNode(const uint nodeIdx, const Vector3f &pos, const Vector3f &dir,
							  const float roughness, const Spectrum &thp) {
		if (mIndex >= nAlloc) {
			atomicAdd(&mInvalidNode, 1u);
			return -1;
		}

		uint idx = atomicAdd(&mIndex, 1u);
		// guard against overflow
		if (mIndex >= nAlloc) {
			mIndex = nAlloc;
			atomicAdd(&mInvalidNode, 1u);
			return -1;
		}

		mThp[idx]		  = thp;
		mPos[idx]		  = pos;
		mDir[idx]		  = dir;
		mL[idx]			  = Spectrum::Zero();
		mLastNodeIdx[idx] = nodeIdx;
		mCost[idx]		  = 0;
		// mNumSamples[idx]  = 0;
		mRoughness[idx] = roughness;
		return (int) idx;
	}

	KRR_CALLABLE uint size() const { return mIndex; }

private:
	uint mInvalidNode; // record the number of invalid node
	uint mIndex;	   // next node to allocate
};

class EARSPixelStateBuffer : public SOA<EARSPixelState> {
public:
	EARSPixelStateBuffer() = default;
	EARSPixelStateBuffer(int n, Allocator alloc) : SOA<EARSPixelState>(n, alloc) {}

	KRR_CALLABLE void setRadiance(int pixelId, Spectrum L) { mL[pixelId] = L; }

	KRR_CALLABLE void addRadiance(int pixelId, Spectrum L) {
		L			= L + Spectrum(mL[pixelId]);
		mL[pixelId] = L;
	}

	KRR_DEVICE void addRadianceAtomic(int pixelId, Spectrum Lval) {
		static_assert(Spectrum::dim == 3, "only support 3 channels");
		atomicAdd(((float *) &mL[pixelId]) + 0, Lval[0]);
		atomicAdd(((float *) &mL[pixelId]) + 1, Lval[1]);
		atomicAdd(((float *) &mL[pixelId]) + 2, Lval[2]);
	}

	KRR_DEVICE void addStatisticAtomic(int pixelId, uint depth, Spectrum Lval) {
		atomicAdd(&mDepth[pixelId], (float) depth);
		addRadianceAtomic(pixelId, Lval);
	}
};

NAMESPACE_END(krr)