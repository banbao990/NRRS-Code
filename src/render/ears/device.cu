#include "render/shared.h"
#include "render/shading.h"
#include "render/simplewpt/workqueue.h"

#include "ears.h"
#include <optix_device.h>

using namespace krr;
NAMESPACE_BEGIN(krr)

#define CLOSEST_RAY_TYPE_USE_IN_DEVICE_CU 0
#define SHADOW_RAY_TYPE_USE_IN_DEVICE_CU 1
#define OPTIX_RAY_TYPES_USE_IN_DEVICE_CU 2

extern "C" __constant__ LaunchParameters<EARSPathTracer> launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle mTraversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(mTraversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						  /* all visible */
			   flags, rayType, OPTIX_RAY_TYPES_USE_IN_DEVICE_CU,  /* ray type and number of types */
			   rayType,											  /* miss SBT index */
			   std::forward<Args>(payload)...); /* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle mTraversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, void *payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(mTraversable, ray, tMax, rayType, flags, u0, u1);
}

KRR_DEVICE_FUNCTION EARSRayWorkItem getRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.mCurrentRayQueue->size());
	return (*launchParams.mCurrentRayQueue)[rayIndex];
}

KRR_DEVICE_FUNCTION EARSShadowRayWorkItem getShadowRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.mShadowRayQueue->size());
	return (*launchParams.mShadowRayQueue)[rayIndex];
}

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo			 = getHitInfo();
	SurfaceInteraction &intr = *getPRD<SurfaceInteraction>();
	EARSRayWorkItem r		 = getRayWorkItem();
	// RGB: ignore lambda
	SampledWavelengths lambda;
	prepareSurfaceInteraction(intr, hitInfo, r.mRay, lambda);

	// push to hit ray queue if mesh has light
	if (intr.light) {
		EARSHitLightWorkItem w = {};

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
		EARSScatterRayWorkItem w = {};

		w.mPixelId = r.mPixelId;
		w.mThp	   = r.mThp;
		w.mIntr	   = intr;
		w.mDepth   = r.mDepth;
		w.mNodeIdx = r.mNodeIdx;

		const int stid = launchParams.mScatterRayQueue->push(w);

		if (launchParams.mEnableRRS || launchParams.mQueryNNCache) {
			// specular case
			bool isSpecular = !(intr.getBsdfType() & BSDF_SMOOTH);
			if (isSpecular) {
				launchParams.mScatterTidQueue->push(stid);
			} else {
				EARSInferenceItem inferenceItem = {};

				inferenceItem.mTid		 = stid;
				inferenceItem.mPos		 = intr.p;
				inferenceItem.mDir		 = intr.wo;
				inferenceItem.mRoughness = intr.sd.roughness;

				launchParams.mNonSpecularTidQueue->push(inferenceItem);
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
	EARSRayWorkItem r		= getRayWorkItem();
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
	if (rayIndex >= launchParams.mShadowRayQueue->size()) return;
	EARSShadowRayWorkItem r = getShadowRayWorkItem();
	uint32_t miss{0};
	traceRay(
		launchParams.mTraversable, r.mRay, r.mMaxT, SHADOW_RAY_TYPE_USE_IN_DEVICE_CU,
		OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		miss);
	if (miss) {
		Spectrum contrib = r.mLi * r.mThp;
		launchParams.mPixelState->addStatisticAtomic(r.mPixelId, r.mDepth, contrib);
		if (launchParams.mEnableTraining) {
			launchParams.mPathState->recordRadiance(r.mNodeIdx, contrib, true);
		}
	}
}

NAMESPACE_END(krr)