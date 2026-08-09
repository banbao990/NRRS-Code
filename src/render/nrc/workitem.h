#pragma once
#include "common.h"
#include "raytracing.h"
#include "render/shared.h"
// inherits all items' definition from the wavefront pathtracer
#include "render/wavefront/workitem.h"
#include "nrcparameters.h"
NAMESPACE_BEGIN(krr)

/* Remember to copy these definitions to workitem.soa whenever changing them. */
struct NRCHitLightWorkItem {
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
	float mA0;
	float mAn;
};

struct NRCRayWorkItem {
	Ray mRay;
	LightSampleContext mCtx;
	float mPdf;
	Spectrum mThp;
	BSDFType mBsdfType;
	uint mDepth;
	uint mPixelId;
	float mA0;
	float mAn;
};

struct NRCScatterRayWorkItem {
	Spectrum mThp;
	SurfaceInteraction mIntr;
	uint mDepth;
	uint mPixelId;
	float mA0;
	float mAn;
	float mAnEle;
};

struct NRCRadianceRecordItem {
	RGB mL;			  // local radiance
	RGB mThp;		  // throughput at current radiance, used to calculate local radiance
	Vector3f mPos;	  // not normalized
	Vector2f mDir;	  // [theta, phi] wi, where theta in [0, pi] and phi in [0, 2pi]
	bool mDelta;	  // is this scatter event is sampled from a delta lobe?
	Vector2f mNormal; // [auxiliary] surface (shading) normal where theta in [0, pi] and phi in [0,
					  // 2pi]
	float mRoughness; // [auxiliary] roughness of the surface
};

// TODO: check the range of each field( as @NRCRadianceRecordItem )
struct NRCInferenceWorkItem {
	uint mPixelId;
	RGB mThp;
	Vector3f mPos; /* not normalized */
	Vector3f mDir; /* normalized */
	float mA0;	   /* initial camera rays */
	float mDepth;
	Vector2f mNormal;
	float mRoughness;
};

struct NRCPathPixelState {
	NRCRadianceRecordItem mRecords[NRC_MAX_TRAIN_DEPTH];
	uint mCurDepth{};
};

struct NRCNetworkInput {
	Vector3f mPos; /* normalized pos to [0, 1]^3 */
	Vector2f mDir; /* normalized dir with 1-norm */
#if NETWORK_AUXILIARY_INPUT
	float mAuxiliary[NRC_DIM_AUXILIARY_INPUT];
#endif
};

// info that used to get the gradients of network output (dL_dy)
struct NRCNetworkOutput {
	RGB mL;
};

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "nrc/include/workitem_soa.h"
#pragma warning(pop)

NAMESPACE_END(krr)