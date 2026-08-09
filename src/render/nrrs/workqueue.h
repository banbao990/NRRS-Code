#pragma once
#include <atomic>

#include "workitem.h"
#include "render/adrrs-nrc/workqueue.h"

#ifdef __INTELLISENSE__
#include <device_atomic_functions.hpp>
#include <sm_20_atomic_functions.hpp>
#endif

NAMESPACE_BEGIN(krr)

class NRRSRayQueue : public WorkQueue<NRRSRayWorkItem> {
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

class NRRSMissRayQueue : public WorkQueue<NRRSMissRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class NRRSHitLightRayQueue : public WorkQueue<NRRSHitLightWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class NRRSShadowRayQueue : public WorkQueue<NRRSShadowRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};

class NRRSScatterRayQueue : public WorkQueue<NRRSScatterRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	inline uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(TidQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((char *) &(m_size)) - ((char *) this));
			Log(Info, "[NRRSScatterRayQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

class NRRSPixelStateBuffer : public SOA<NRRSPixelState> {
public:
	NRRSPixelStateBuffer() = default;
	NRRSPixelStateBuffer(int n, Allocator alloc) : SOA<NRRSPixelState>(n, alloc) {}

	KRR_CALLABLE void setRadiance(int pixelId, Spectrum L_val) { mL[pixelId] = L_val; }
	KRR_CALLABLE void addRadiance(int pixelId, Spectrum L_val) {
		L_val		= L_val + Spectrum(mL[pixelId]);
		mL[pixelId] = L_val;
	}
	KRR_DEVICE void addRadianceAtomic(int pixelId, Spectrum L_val) {
		atomicAdd(((float *) &mL[pixelId]) + 0, L_val.x());
		atomicAdd(((float *) &mL[pixelId]) + 1, L_val.y());
		atomicAdd(((float *) &mL[pixelId]) + 2, L_val.z());
	}
};

class NRRSInferenceQueue : public WorkQueue<NRRSInferenceWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(NRRSInferenceQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((byte *) &(this->m_size)) - ((byte *) this));
			Log(Info, "[NRRSInferenceQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

class NRRSPathNodesBuffer : public SOA<NRRSRadianceRecordItem> {
public:
	NRRSPathNodesBuffer() = default;
	NRRSPathNodesBuffer(int n, Allocator alloc) : SOA<NRRSRadianceRecordItem>(n, alloc) {}

	KRR_DEVICE void reset(const bool log) {
		if (log && mInvalidNode > 0) {
			printf("[Nodes] Last Statistic: mIndex = %d, mInvalidNode = %d\n", mIndex,
				   mInvalidNode);
		}
		mIndex		 = 0;
		mInvalidNode = 0;
	}

	KRR_DEVICE void recordRadiance(uint nodeIdx, const Spectrum &L) {
		static_assert(Spectrum::dim == 3, "only support 3 channels");
		if (nodeIdx >= mIndex || nodeIdx == -1) {
			return;
		}
		while (nodeIdx != -1) {
			// record the radiance
			atomicAdd(((float *) &(mL[nodeIdx])) + 0, L.x());
			atomicAdd(((float *) &(mL[nodeIdx])) + 1, L.y());
			atomicAdd(((float *) &(mL[nodeIdx])) + 2, L.z());
			nodeIdx = mLastNodeIdx[nodeIdx];
		}
	}

	// KRR_DEVICE void recordRRS(const uint nodeIdx, const float rrs) {
	//	if (nodeIdx != -1) {
	//		mRRS[nodeIdx] = rrs;
	//	}
	// }

	KRR_DEVICE int recordNode(const uint nodeIdx, const Vector3f &pos, const Vector3f &dir,
							  const float roughness, const Spectrum &thp, const uint pixelId,
							  const float rrs, const float depth) {
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
		mDir[idx]		  = utils::worldToLatLong(dir);
		mRoughness[idx]	  = roughness;
		mL[idx]			  = Spectrum::Zero();
		mLastNodeIdx[idx] = nodeIdx;
		mPixelId[idx]	  = pixelId;
		mRRS[idx]		  = rrs;
		mDepth[idx]		  = depth;
		return (int) idx;
	}

	KRR_CALLABLE uint size() const { return mIndex; }

	uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(NRRSInferenceQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((byte *) &(this->mIndex)) - ((byte *) this));
			Log(Info, "[NRRSPathNodesBuffer] offset of size: %d", offset);
		}
		return offset;
	}

private:
	uint mInvalidNode; // record the number of invalid node
	uint mIndex;	   // next node to allocate
};

NAMESPACE_END(krr)