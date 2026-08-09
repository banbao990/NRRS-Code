#include "nrcparameters.h"
#include "nrcguided.h"
#include "nrc.h"
#include "render/shading.h"
#include "render/shared.h"

#include <optix_device.h>

using namespace krr;
NAMESPACE_BEGIN(krr)

extern "C" __constant__ LaunchParameters<NRCPathTracer> launchParams;

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

KRR_DEVICE_FUNCTION NRCRayWorkItem getRayWorkItem() {
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
	NRCRayWorkItem r		 = getRayWorkItem();
	// RGB: ignore lambda
	SampledWavelengths lambda;
	prepareSurfaceInteraction(intr, hitInfo, r.mRay, lambda);

	if (intr.light) { // push to hit ray queue if mesh has light
		NRCHitLightWorkItem w = {};
		w.mLight			  = intr.light;
		w.mCtx				  = r.mCtx;
		w.mPos				  = intr.p;
		w.mNormal			  = intr.n;
		w.mWo				  = intr.wo;
		w.mUv				  = intr.uv;
		w.mDepth			  = r.mDepth;
		w.mPixelId			  = r.mPixelId;
		w.mThp				  = r.mThp;
		w.mPdf				  = r.mPdf;
		w.mBsdfType			  = r.mBsdfType;
		launchParams.mHitLightRayQueue->push(w);
	}

	// process material and push to material evaluation queue (if eligible)
	if (any(r.mThp)) {
		NRCScatterRayWorkItem w = {};
		w.mPixelId				= r.mPixelId;
		w.mThp					= r.mThp;
		w.mIntr					= intr;
		w.mDepth				= r.mDepth;

		// update nrc termination heuristic
		w.mAnEle = (intr.p - r.mRay.origin).squaredNorm() / AbsDot(intr.n, intr.wo);
		if (r.mDepth == 0) {
			// only update once, when depth = 0 (camera/primary ray)
			w.mA0 = w.mAnEle * M_PI_4;
		} else {
			w.mA0 = r.mA0;
		}
		launchParams.mScatterRayQueue->push(w);
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
	NRCRayWorkItem r = getRayWorkItem();
	SurfaceInteraction intr;
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
	uint32_t miss{0};
	traceRay(
		launchParams.mTraversable, r.mRay, r.mMaxT, 1,
		OptixRayFlags(OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		miss);
	if (miss) {
		Spectrum contrib = r.mLi * r.mThp;
		if (!launchParams.isTraining) {
			launchParams.mPixelState->addRadiance(r.mPixelId, contrib);
		} else {
			launchParams.guidedState->recordRadiance(r.mPixelId, contrib);
		}
	}
}

NAMESPACE_END(krr)