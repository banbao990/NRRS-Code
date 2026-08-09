#pragma once
#include "raytracing.h"
#include "render/wavefront/workitem.h"
#include "render/common/commonitem.h"

NAMESPACE_BEGIN(krr)

// copy to soa

struct EARSInferenceItem {
	uint mTid;
	Vector3f mPos;
	Vector3f mDir;
	float mRoughness;
};

struct EARSRadianceRecordItem {
	// local radiance
	Spectrum mL;
	// throughput at current radiance, used to calculate local radiance [not multiplied by BSDF]
	Spectrum mThp;
	// not normalized
	Vector3f mPos;
	// normalized
	Vector3f mDir;
	// last node index, default -1 (first node)
	int mLastNodeIdx;
	// computational cost(infact equals to the whole path's length - the node's depth)
	uint mCost;
	// how many samples have been accumulated
	// uint mNumSamples;
	// roughness
	float mRoughness;
};

struct EARSRayWorkItem {
	Ray mRay;
	LightSampleContext mCtx;
	float mPdf;
	Spectrum mThp;
	BSDFType mBsdfType;
	uint mDepth;
	uint mPixelId;
	int mNodeIdx; // last node
};

typedef EARSRayWorkItem EARSMissRayWorkItem;

struct EARSHitLightWorkItem {
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
	int mNodeIdx; // last node
};

struct EARSShadowRayWorkItem {
	Ray mRay;
	float mMaxT;
	Spectrum mLi;
	Spectrum mThp;
	uint mDepth;
	uint mPixelId;
	int mNodeIdx; // last node
};

struct EARSScatterRayWorkItem {
	Spectrum mThp;
	int mNodeIdx;
	SurfaceInteraction mIntr;
	uint mDepth;
	uint mPixelId;
};

struct EARSPixelState {
	Spectrum mL;
	float mDepth;
	float mDiff2; // (r2+g2+r2)/3
	PCGSampler mSampler;
};

#include "ears/include/workitem_soa.h"
#include "common/include/commonitem_soa.h"

NAMESPACE_END(krr)