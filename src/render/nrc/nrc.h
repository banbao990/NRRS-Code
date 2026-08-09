#pragma once

#include "device/optix.h"
#include "render/bsdf.h"
#include "render/lightsampler.h"
#include "workqueue.h"
#include "sampler.h"
#include "scene.h"
#include "window.h"

NAMESPACE_BEGIN(krr)

class NRCPathTracer;

template <> struct LaunchParameters<NRCPathTracer> {
	NRCRayQueue *mCurrentRayQueue;
	NRCRayQueue *mNextRayQueue;
	SWPTShadowRayQueue *mShadowRayQueue;
	NRCMissRayQueue *mMissRayQueue;
	NRCHitLightRayQueue *mHitLightRayQueue;
	NRCScatterRayQueue *mScatterRayQueue;
	bool isTraining;

	SWPTPixelStateBuffer *mPixelState;
	NRCPathPixelStateBuffer *guidedState;
	NRCTrainState trainState;
	rt::SceneData mSceneData;
	OptixTraversableHandle mTraversable;
};

NAMESPACE_END(krr)