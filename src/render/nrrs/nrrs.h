#pragma once

#include "device/optix.h"
#include "render/bsdf.h"
#include "render/lightsampler.h"
#include "sampler.h"
#include "scene.h"
#include "window.h"
#include "util/film.h"

#include "workitem.h"
#include "render/common/commonworkqueue.h"
#include "render/ears/workqueue.h"
#include "workqueue.h"
#include "render/nrc/nrcguided.h"

NAMESPACE_BEGIN(krr)

class MyFilm;

class NRRSPathTracer;
template <> struct LaunchParameters<NRRSPathTracer> {
	NRRSRayQueue *mCurrentRayQueue;
	NRRSRayQueue *mNextRayQueue;
	NRRSShadowRayQueue *mShadowRayQueue;
	NRRSMissRayQueue *mMissRayQueue;
	NRRSHitLightRayQueue *mHitLightRayQueue;
	NRRSScatterRayQueue *mScatterRayQueue;
	NRRSInferenceQueue *mInferenceQueue;
	TidQueue *mScatterTidQueue;
	bool mIsTraining;
	bool mUseRRS;
	bool mShowLi; // set true iff showLi & depth condition is true

	NRRSPixelStateBuffer *mPixelState;
	NRRSPathNodesBuffer *mPathState;
	rt::SceneData mSceneData;
	OptixTraversableHandle mTraversable;
	// MyFilm *mRenderedImage;
	float *mSpecularBuffer;
};

NAMESPACE_END(krr)