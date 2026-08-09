#pragma once

#include "device/optix.h"
#include "render/bsdf.h"
#include "render/lightsampler.h"
#include "render/nrc/nrcguided.h"
#include "workqueue.h"
#include "sampler.h"
#include "scene.h"
#include "window.h"

NAMESPACE_BEGIN(krr)

class ADNPathTracer;
template <> struct LaunchParameters<ADNPathTracer> {
	SWPTRayQueue *mCurrentRayQueue;
	SWPTRayQueue *mNextRayQueue;
	SWPTShadowRayQueue *mShadowRayQueue;
	SWPTMissRayQueue *mMissRayQueue;
	SWPTHitLightRayQueue *mHitLightRayQueue;
	SWPTScatterRayQueue *mScatterRayQueue;
	ADNInferenceQueue *mInferenceQueue;
	TidQueue *mScatterTidQueue;
	bool mIsTraining;
	bool mShowLi;
	int mShowLiDepth;

	SWPTPixelStateBuffer *mPixelState;
	NRCPathPixelStateBuffer *mPathState;
	rt::SceneData mSceneData;
	OptixTraversableHandle mTraversable;
};

NAMESPACE_END(krr)