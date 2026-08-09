#pragma once
#include "common.h"
#include "raytracing.h"
#include "device/soa.h"
#include "render/shared.h"
#include "render/spectrum.h"
#include "render/materials/bxdf.h"

NAMESPACE_BEGIN(krr)

using namespace rt;

/* Remember to copy these definitions to workitem.soa whenever changing them. */

struct SWPTPixelState {
	Spectrum mL;
	PCGSampler mSampler;
};

struct SWPTRayWorkItem {
	Ray mRay;
	LightSampleContext mCtx;
	float mPdf;
	Spectrum mThp;
	BSDFType mBsdfType;
	uint mDepth;
	uint mPixelId;
};

typedef SWPTRayWorkItem SWPTMissRayWorkItem;

struct SWPTHitLightWorkItem {
	Light mLight;
	LightSampleContext mCtx;
	float mPdf;
	Vector3f mPos;
	Vector3f mWo;
	Vector3f mNormal;
	Vector2f mUv;
	Spectrum mThp;
	BSDFType mBsdfType;
	uint mDepth;
	uint mPixelId;
};

struct SWPTShadowRayWorkItem {
	Ray mRay;
	float mMaxT;
	Spectrum mLi;
	Spectrum mThp;
	uint mDepth;
	uint mPixelId;
};

struct SWPTScatterRayWorkItem {
	Spectrum mThp;
	SurfaceInteraction mIntr;
	uint mDepth;
	uint mPixelId;
};

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "render/wavefront/basic_soa.h"
#include "simplewpt/include/workitem_soa.h"
#pragma warning(pop)

NAMESPACE_END(krr)