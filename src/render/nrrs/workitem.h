#pragma once

#include "render/adrrs-nrc/workitem.h"

NAMESPACE_BEGIN(krr)

/* Remember to copy these definitions to workitem.soa whenever changing them. */
// start
struct NRRSInferenceWorkItem {
	uint mScatterQueueIndex;
	Vector3f mPos;	  // not normalized
	Vector2f mDir;	  // normalized
	float mRoughness; // roughness
	RGB mThp;		  // original throughput, before rrs
};

struct NRRSRadianceRecordItem {
	// local radiance
	Spectrum mL;
	// throughput at current radiance, used to calculate local radiance [not multiplied by BSDF]
	Spectrum mThp;
	// not normalized
	Vector3f mPos;
	// normalized
	Vector2f mDir;
	// roughness
	float mRoughness;
	// last node index, default -1 (first node)
	int mLastNodeIdx;
	// pixel id
	uint mPixelId;
	// RRS after normalization
	float mRRS;
	// depth
	uint mDepth;
};

struct NRRSPixelState {
	Spectrum mL;
	uint mTrainingSamples; // the total training samples in this pixel
						   // [< mNumberSamples, may discard]
	PCGSampler mSampler;
};

struct NRRSRayWorkItem {
	Ray mRay;
	LightSampleContext mCtx;
	float mPdf;
	Spectrum mThp;
	BSDFType mBsdfType;
	uint mDepth;
	uint mPixelId;
	int mNodeIdx; // last node
};

typedef NRRSRayWorkItem NRRSMissRayWorkItem;

struct NRRSHitLightWorkItem {
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

struct NRRSShadowRayWorkItem {
	Ray mRay;
	float mMaxT;
	Spectrum mLi;
	Spectrum mThp;
	uint mDepth;
	uint mPixelId;
	int mNodeIdx; // last node
};

struct NRRSScatterRayWorkItem {
	Spectrum mThp;
	int mNodeIdx;
	SurfaceInteraction mIntr;
	uint mDepth;
	uint mPixelId;
	float mRRS;
};

// end

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "nrrs/include/workitem_soa.h"
#pragma warning(pop)

NAMESPACE_END(krr)