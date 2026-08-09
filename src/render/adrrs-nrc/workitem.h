#pragma once
#include "common.h"
#include "raytracing.h"
#include "render/shared.h"

// inherits all items' definition from the wavefront pathtracer
#include "render/wavefront/workitem.h"
// inherits all items' definition from the NRC pathtracer
// #include "render/nrc/workitem.h"

#include "render/common/commonitem.h"
NAMESPACE_BEGIN(krr)

/* Remember to copy these definitions to workitem.soa whenever changing them. */
// start
struct ADNInferenceWorkItem {
	uint mScatterQueueIndex;
	Vector3f mPos; /* not normalized */
	Vector2f mDir; /* normalized */
	bool mIsSpecular;
	Vector2f mNormal; /* normalized */
	float mRoughness;
};
// end

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "adrrs-nrc/include/workitem_soa.h"
#include "common/include/commonitem_soa.h"
#pragma warning(pop)

NAMESPACE_END(krr)