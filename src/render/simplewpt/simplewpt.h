#pragma once

#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"
#include "device/optix.h"
#include "workqueue.h"

NAMESPACE_BEGIN(krr)

class SWPTPixelStateBuffer;
class SimpleWavefrontPathTracer;

template <> struct LaunchParameters<SimpleWavefrontPathTracer> {
	SWPTRayQueue *mCurrentRayQueue;
	SWPTRayQueue *mNextRayQueue;
	SWPTShadowRayQueue *mShadowRayQueue;
	SWPTMissRayQueue *mMissRayQueue;
	SWPTHitLightRayQueue *mHitLightRayQueue;
	SWPTScatterRayQueue *mScatterRayQueue;

	SWPTPixelStateBuffer *mPixelState;
	rt::SceneData mSceneData;
	OptixTraversableHandle mTraversable;
};

NAMESPACE_END(krr)