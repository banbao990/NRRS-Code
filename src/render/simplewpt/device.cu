#include "render/shared.h"
#include "render/shading.h"
#include "simplewpt.h"
#include "workqueue.h"

#include <optix_device.h>

NAMESPACE_BEGIN(krr)

extern "C" __constant__ LaunchParameters<SimpleWavefrontPathTracer> launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle mTraversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(mTraversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						  /* all visible */
			   flags, rayType, 2,								  /* ray type and number of types */
			   rayType,											  /* miss SBT index */
			   std::forward<Args>(payload)...); /* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle mTraversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, void *payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(mTraversable, ray, tMax, rayType, flags, u0, u1);
}

KRR_DEVICE_FUNCTION SWPTRayWorkItem getRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.mCurrentRayQueue->size());
	return (*launchParams.mCurrentRayQueue)[rayIndex];
}

KRR_DEVICE_FUNCTION SWPTShadowRayWorkItem getShadowRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.mShadowRayQueue->size());
	return (*launchParams.mShadowRayQueue)[rayIndex];
}

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo			 = getHitInfo();
	SurfaceInteraction &intr = *getPRD<SurfaceInteraction>();
	SWPTRayWorkItem r		 = getRayWorkItem();
	// RGB: ignore lambda
	SampledWavelengths lambda;
	prepareSurfaceInteraction(intr, hitInfo, r.mRay, lambda);

	// push to hit ray queue if mesh has light
	if (intr.light) {
		SWPTHitLightWorkItem w = {};
		w.mLight			   = intr.light;
		w.mCtx				   = r.mCtx;
		w.mPos				   = intr.p;
		w.mNormal			   = intr.n;
		w.mWo				   = intr.wo;
		w.mUv				   = intr.uv;
		w.mDepth			   = r.mDepth;
		w.mPixelId			   = r.mPixelId;
		w.mThp				   = r.mThp;
		w.mPdf				   = r.mPdf;
		w.mBsdfType			   = r.mBsdfType;
		launchParams.mHitLightRayQueue->push(w);
	}

	// process material and push to material evaluation queue (if eligible)
	if (any(r.mThp)) {
		SWPTScatterRayWorkItem w = {};
		w.mPixelId				 = r.mPixelId;
		w.mThp					 = r.mThp;
		w.mIntr					 = intr;
		launchParams.mScatterRayQueue->push(w);
		w.mDepth = r.mDepth;
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
	SWPTRayWorkItem r		= getRayWorkItem();
	SurfaceInteraction intr = {};
	traceRay(launchParams.mTraversable, r.mRay, M_FLOAT_INF, 0, OPTIX_RAY_FLAG_NONE,
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

	SWPTShadowRayWorkItem r = getShadowRayWorkItem();
	uint32_t visible{0};
	traceRay(
		launchParams.mTraversable, r.mRay, r.mMaxT, 1,
		OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		visible);
	if (visible) {
		launchParams.mPixelState->addRadiance(r.mPixelId, r.mLi * r.mThp);
	}
}

NAMESPACE_END(krr)