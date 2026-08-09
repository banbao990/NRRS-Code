#pragma once
#include "render/wavefront/wavefront.h"
#include "workqueue.h"
#include "render/common/commonworkqueue.h"

NAMESPACE_BEGIN(krr)

class EARSPathTracer;
template <> struct LaunchParameters<EARSPathTracer> {
	EARSRayQueue *mCurrentRayQueue;
	EARSRayQueue *mNextRayQueue;
	EARSShadowRayQueue *mShadowRayQueue;
	EARSMissRayQueue *mMissRayQueue;
	EARSHitLightRayQueue *mHitLightRayQueue;
	EARSScatterRayQueue *mScatterRayQueue;

	TidQueue *mScatterTidQueue;
	EARSInferenceQueue *mNonSpecularTidQueue;
	bool mEnableTraining;
	bool mEnableRRS;
	bool mGBufferEnableSpecularContinue;
	RGB *mAlbedoBuffer;
	RGB *mNormalBuffer;
	float mGBufferSppInv;
	bool mQueryNNCache; // set true iff queryNN & depth condition is true

	EARSPixelStateBuffer *mPixelState;
	rt::SceneData mSceneData;
	EARSPathNodesBuffer *mPathState;
	OptixTraversableHandle mTraversable;
};

struct EARSImageStatistic {
public:
	KRR_DEVICE void reset() {
		if (mSpp == 0) {
			mEARSFactor = 1.0f;
		} else {
			mEARSFactor = mCost / mSquareError;
		}
		mSquareError = 0.0f;
		mCost		 = 0.0f;
		mSpp		 = 0;
	}

	KRR_DEVICE void record(const float squareError, const float cost) {
		mSquareError += squareError;
		mCost += cost;
		mSpp++;
	}

	KRR_DEVICE float getCost() const {
		float ret = (mSpp > 0) ? mCost / mSpp : 0.0f;
		return ret;
	}
	KRR_DEVICE float getSquareError() const {
		float ret = (mSpp > 0) ? mSquareError / mSpp : 0.0f;
		return ret;
	}
	KRR_DEVICE float getEARSFactor() const { return mEARSFactor; }
	KRR_DEVICE float getEARSFactorCurrent() const {
		float ret = (mSpp > 0) ? mCost / mSquareError : 1.0f;
		return ret;
	}

private:
	float mSquareError;
	float mCost;
	uint mSpp;
	float mEARSFactor;
};

typedef struct {
	float *mData;
} FloatPointerWarpper;

NAMESPACE_END(krr)