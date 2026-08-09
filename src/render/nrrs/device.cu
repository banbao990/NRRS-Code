#include "nrrsparams.h"
#include "nrrs.h"
#include "render/shading.h"
#include "render/shared.h"

#include <optix_device.h>

NAMESPACE_BEGIN(krr)

extern "C" __constant__ LaunchParameters<NRRSPathTracer> launchParams;

#define RAY_TYPES 4
#define CLOSEST_RAY 0
#define SHODOW_RAY 1
#define CLOSET_RAY_SIMPLE 2
#define SHADOW_RAY_SIMPLE 3

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle mTraversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(mTraversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						  /* all visible */
			   flags, rayType, RAY_TYPES,						  /* ray type and number of types */
			   rayType,											  /* miss SBT index */
			   std::forward<Args>(payload)...); /* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle mTraversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, void *payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(mTraversable, ray, tMax, rayType, flags, u0, u1);
}

KRR_DEVICE_FUNCTION NRRSRayWorkItem getRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.mCurrentRayQueue->size());
	return (*launchParams.mCurrentRayQueue)[rayIndex];
}

KRR_DEVICE_FUNCTION NRRSShadowRayWorkItem getShadowRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.mShadowRayQueue->size());
	return (*launchParams.mShadowRayQueue)[rayIndex];
}

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo			 = getHitInfo();
	SurfaceInteraction &intr = *getPRD<SurfaceInteraction>();
	NRRSRayWorkItem r		 = getRayWorkItem();
	// RGB: ignore lambda
	SampledWavelengths lambda;
	prepareSurfaceInteraction(intr, hitInfo, r.mRay, lambda);

	// push to hit ray queue if mesh has light
	if (intr.light) {
		NRRSHitLightWorkItem w = {};

		w.mLight	= intr.light;
		w.mCtx		= r.mCtx;
		w.mPos		= intr.p;
		w.mNormal	= intr.n;
		w.mWo		= intr.wo;
		w.mUv		= intr.uv;
		w.mDepth	= r.mDepth;
		w.mPixelId	= r.mPixelId;
		w.mThp		= r.mThp;
		w.mPdf		= r.mPdf;
		w.mBsdfType = r.mBsdfType;
		w.mNodeIdx	= r.mNodeIdx;
		launchParams.mHitLightRayQueue->push(w);
	}
	// process material and push to material evaluation queue (if eligible)
	if (any(r.mThp)) {
		NRRSScatterRayWorkItem w = {};

		w.mPixelId = r.mPixelId;
		w.mThp	   = r.mThp;
		w.mIntr	   = intr;
		w.mDepth   = r.mDepth;
		w.mNodeIdx = r.mNodeIdx;
		w.mRRS	   = 1.0f; // default value

		int scatterQueueIndex = launchParams.mScatterRayQueue->push(w);

		if (launchParams.mUseRRS || launchParams.mShowLi) {
			// means use RRS / showLi enabled

			// specular case: current hit point is specular
			bool isSpecular = !(intr.getBsdfType() & BSDF_SMOOTH);
			if (isSpecular) {
				launchParams.mScatterTidQueue->push(scatterQueueIndex);
				if (launchParams.mSpecularBuffer) {
					atomicAdd(launchParams.mSpecularBuffer + r.mPixelId, 1.0f);
				}
			} else {
				NRRSInferenceWorkItem iitem = {};
				iitem.mScatterQueueIndex	= scatterQueueIndex;
				// view direction
				iitem.mPos = intr.p;
				iitem.mDir = utils::worldToLatLong(intr.wo);
				// roughness
				iitem.mRoughness = intr.sd.roughness;
				// relative throughput
				// iitem.mThp = r.mThp / RGB(launchParams.mRenderedImage->getPixel(r.mPixelId));
				iitem.mThp = r.mThp;
				launchParams.mInferenceQueue->push(iitem);
			}
		}
	}
}

extern "C" __global__ void KRR_RT_AH(Closest)() {
	if (alphaKilled(getHitInfo())) {
		optixIgnoreIntersection();
	}
}

extern "C" __global__ void KRR_RT_MS(Closest)() {
	launchParams.mMissRayQueue->push(getRayWorkItem());
}

extern "C" __global__ void KRR_RT_RG(Closest)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.mCurrentRayQueue->size()) {
		return;
	}
	NRRSRayWorkItem r		= getRayWorkItem();
	SurfaceInteraction intr = {};
	traceRay(launchParams.mTraversable, r.mRay, M_FLOAT_INF, CLOSEST_RAY, OPTIX_RAY_FLAG_NONE,
			 (void *) &intr);
}

extern "C" __global__ void KRR_RT_AH(Shadow)() {
	if (alphaKilled(getHitInfo())) {
		optixIgnoreIntersection();
	}
}

extern "C" __global__ void KRR_RT_MS(Shadow)() { optixSetPayload_0(1); }

extern "C" __global__ void KRR_RT_RG(Shadow)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.mShadowRayQueue->size()) {
		return;
	}
	NRRSShadowRayWorkItem r = getShadowRayWorkItem();
	uint32_t miss{0};
	traceRay(
		launchParams.mTraversable, r.mRay, r.mMaxT, SHODOW_RAY,
		OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		miss);
	if (miss) {
		Spectrum contrib = r.mLi * r.mThp;
		if (launchParams.mIsTraining) {
			launchParams.mPathState->recordRadiance(r.mNodeIdx, contrib);
		}
		launchParams.mPixelState->addRadianceAtomic(r.mPixelId, contrib);
	}
}

// simple render mode

KRR_DEVICE_FUNCTION SWPTRayWorkItem getRayWorkItemSimple() {
	int rayIndex(optixGetLaunchIndex().x);
	const SWPTRayQueue *queue = (SWPTRayQueue *) launchParams.mCurrentRayQueue;
	DCHECK_LT(rayIndex, queue->size());
	return (*queue)[rayIndex];
}

KRR_DEVICE_FUNCTION SWPTShadowRayWorkItem getShadowRayWorkItemSimple() {
	int rayIndex(optixGetLaunchIndex().x);
	const SWPTShadowRayQueue *queue = (SWPTShadowRayQueue *) launchParams.mShadowRayQueue;
	DCHECK_LT(rayIndex, queue->size());
	return (*queue)[rayIndex];
}

extern "C" __global__ void KRR_RT_CH(ClosestSimple)() {
	HitInfo hitInfo			 = getHitInfo();
	SurfaceInteraction &intr = *getPRD<SurfaceInteraction>();
	SWPTRayWorkItem r		 = getRayWorkItemSimple();
	// RGB: ignore lambda
	SampledWavelengths lambda;
	prepareSurfaceInteraction(intr, hitInfo, r.mRay, lambda);

	// push to hit ray queue if mesh has light
	if (intr.light) {
		SWPTHitLightWorkItem w = {};

		w.mLight	= intr.light;
		w.mCtx		= r.mCtx;
		w.mPos		= intr.p;
		w.mNormal	= intr.n;
		w.mWo		= intr.wo;
		w.mUv		= intr.uv;
		w.mDepth	= r.mDepth;
		w.mPixelId	= r.mPixelId;
		w.mThp		= r.mThp;
		w.mPdf		= r.mPdf;
		w.mBsdfType = r.mBsdfType;
		((SWPTHitLightRayQueue *) launchParams.mHitLightRayQueue)->push(w);
	}
	// process material and push to material evaluation queue (if eligible)
	if (any(r.mThp)) {
		SWPTScatterRayWorkItem w = {};

		w.mPixelId = r.mPixelId;
		w.mThp	   = r.mThp;
		w.mIntr	   = intr;
		w.mDepth   = r.mDepth;

		int scatterQueueIndex = ((SWPTScatterRayQueue *) launchParams.mScatterRayQueue)->push(w);

		// specular case
		bool isSpecular = !(intr.getBsdfType() & BSDF_SMOOTH); // current hit point is specular
		if (isSpecular) {
			launchParams.mScatterTidQueue->push(scatterQueueIndex);
		} else {
			NRRSInferenceWorkItem iitem = {};
			iitem.mScatterQueueIndex	= scatterQueueIndex;
			// view direction
			iitem.mPos = intr.p;
			iitem.mDir = utils::worldToLatLong(normalize(intr.wo));
			// roughness
			iitem.mRoughness = intr.sd.roughness;
			// relative throughput
			// iitem.mThp = r.mThp / RGB(launchParams.mRenderedImage->getPixel(r.mPixelId));
			iitem.mThp = r.mThp;
			launchParams.mInferenceQueue->push(iitem);
		}
	}
}

extern "C" __global__ void KRR_RT_AH(ClosestSimple)() {
	if (alphaKilled(getHitInfo())) {
		optixIgnoreIntersection();
	}
}

extern "C" __global__ void KRR_RT_MS(ClosestSimple)() {
	((SWPTMissRayQueue *) launchParams.mMissRayQueue)->push(getRayWorkItemSimple());
}

extern "C" __global__ void KRR_RT_RG(ClosestSimple)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= ((SWPTRayQueue *) launchParams.mCurrentRayQueue)->size()) {
		return;
	}
	SWPTRayWorkItem r		= getRayWorkItemSimple();
	SurfaceInteraction intr = {};
	traceRay(launchParams.mTraversable, r.mRay, M_FLOAT_INF, CLOSET_RAY_SIMPLE, OPTIX_RAY_FLAG_NONE,
			 (void *) &intr);
}

extern "C" __global__ void KRR_RT_AH(ShadowSimple)() {
	if (alphaKilled(getHitInfo())) {
		optixIgnoreIntersection();
	}
}

extern "C" __global__ void KRR_RT_MS(ShadowSimple)() { optixSetPayload_0(1); }

extern "C" __global__ void KRR_RT_RG(ShadowSimple)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= ((SWPTShadowRayQueue *) launchParams.mShadowRayQueue)->size()) {
		return;
	}
	SWPTShadowRayWorkItem r = getShadowRayWorkItemSimple();
	uint32_t miss{0};
	traceRay(
		launchParams.mTraversable, r.mRay, r.mMaxT, SHADOW_RAY_SIMPLE,
		OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		miss);
	if (miss) {
		Spectrum contrib = r.mLi * r.mThp;
		launchParams.mPixelState->addRadianceAtomic(r.mPixelId, contrib);
	}
}

NAMESPACE_END(krr)