#include "render/shared.h"
#include "render/shading.h"
#include "render/simplewpt/workqueue.h"

#include "denoisetask.h"
#include <optix_device.h>

using namespace krr;
NAMESPACE_BEGIN(krr)

#define GBUFFER_RAY_TYPE_USE_IN_DEVICE_CU 0
#define OPTIX_RAY_TYPES_USE_IN_DEVICE_CU 1

extern "C" __constant__ LaunchParameters<DenoiseTask> launchParams;

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

typedef struct {
	SurfaceInteraction mIntr;
	bool mOK;
	float mThp; // here the throughput is irrelevant with the albedo and normal
	uint mPixelId;
	Ray mRay;
} SurfaceInteractionWrapper;

KRR_DEVICE_FUNCTION void setAlbedoAndNormal(uint idx, RGB albedoVal, RGB normalVal) {
	RGB *const albedo = launchParams.mAlbedoBuffer;
	RGB *const normal = launchParams.mNormalBuffer;
	// no atomic is needed
	albedo[idx] += albedoVal * launchParams.mGBufferSppInv;
	normal[idx] += normalVal * launchParams.mGBufferSppInv;
}

extern "C" __global__ void KRR_RT_CH(GBuffer)() {
	HitInfo hitInfo						   = getHitInfo();
	SurfaceInteractionWrapper &intrWrapper = *getPRD<SurfaceInteractionWrapper>();
	SurfaceInteraction &intr			   = intrWrapper.mIntr;
	SampledWavelengths lambda;
	prepareSurfaceInteraction(intr, hitInfo, intrWrapper.mRay, lambda);

	const BSDFType bsdfType = intr.getBsdfType();
	// if specular, recursive ray tracing
	if (launchParams.mGBufferEnableSpecularContinue && (BSDF_SPECULAR & bsdfType) &&
		(!intrWrapper.mOK)) {
		return;
	} else {
		RGB albedo = intrWrapper.mThp * intr.sd.diffuse;
		RGB normal = intrWrapper.mThp * (RGB) intr.n; // shading normal
		setAlbedoAndNormal(intrWrapper.mPixelId, albedo, normal);
		intrWrapper.mOK = true;
	}
}

extern "C" __global__ void KRR_RT_AH(GBuffer)() {
	if (alphaKilled(getHitInfo())) {
		optixIgnoreIntersection();
	}
}

extern "C" __global__ void KRR_RT_MS(GBuffer)() {
	SurfaceInteractionWrapper &intrWrapper = *getPRD<SurfaceInteractionWrapper>();
	intrWrapper.mOK						   = true;
	float thp							   = intrWrapper.mThp;

	setAlbedoAndNormal(intrWrapper.mPixelId, thp * RGB(1.0f, 1.0f, 1.0f),
					   thp * RGB(0.0f, 1.0f, 0.0f));
}

extern "C" __global__ void KRR_RT_RG(GBuffer)() {
	Vector3ui launchIndex = optixGetLaunchIndex();
	Vector2ui pixel		  = {launchIndex[0], launchIndex[1]};

	const uint frameID	   = launchParams.mDenoiseFrameId;
	const uint32_t pixelId = pixel[0] + pixel[1] * launchParams.mFrameBufferSize[0];

	PCGSampler samplerInner;
	samplerInner.setPixelSample(pixel, frameID);
	samplerInner.advance(pixelId * 256);
	Sampler sampler = &samplerInner;

	for (int sppIndex = 0; sppIndex < launchParams.mGBufferSpp; ++sppIndex) {
		SurfaceInteractionWrapper intrWrapper = {};
		SurfaceInteraction &intr			  = intrWrapper.mIntr;
		intrWrapper.mThp					  = 1.0f;
		intrWrapper.mPixelId				  = pixelId;
		intrWrapper.mRay =
			launchParams.mCamera.getRay(pixel, launchParams.mFrameBufferSize, sampler);

		const int maxDepth = 10;
		for (int depth = 0; depth <= maxDepth; ++depth) {
			intrWrapper.mOK = (depth == maxDepth);

			traceRay(launchParams.mTraversable, intrWrapper.mRay, M_FLOAT_INF,
					 GBUFFER_RAY_TYPE_USE_IN_DEVICE_CU, OPTIX_RAY_FLAG_NONE, (void *) &intrWrapper);
			if (intrWrapper.mOK) {
				break;
			}

			// generate scatter ray
			BSDFType bsdfType = intr.getBsdfType();
			Vector3f woLocal  = intr.toLocal(intr.wo);
			BSDFSample sample = BxDF::sample(intr, woLocal, sampler, (int) bsdfType);
			intrWrapper.mThp *= sample.pdf;
			Vector3f wiWorld = intr.toWorld(sample.wi);
			Vector3f p		 = offsetRayOrigin(intr.p, intr.n, wiWorld);

			// set ray item
			intrWrapper.mRay = {p, wiWorld};
		}
	}
}

NAMESPACE_END(krr)