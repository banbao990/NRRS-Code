#pragma once
#include <atomic>
#include <array>
#include <stack>

#include "common.h"
#include "sampler.h"
#include "logger.h"
#include "device/gpustd.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/atomic.h"

#include "util/check.h"

NAMESPACE_BEGIN(krr)

struct SimpleTreeNode {
	float mVal;

	KRR_HOST void store(const float val) { mVal = val; }
	KRR_HOST void store(const SimpleTreeNode &node) { mVal = node.mVal; }

	KRR_CALLABLE float load() const { return mVal; }
	KRR_CALLABLE SimpleTreeNode loadAll() const { return *this; }

	KRR_DEVICE void add(const SimpleTreeNode val) {
#ifdef KRR_DEVICE_CODE
		atomicAdd((float *) &mVal, val.mVal);
#else
		Log(Fatal, "SimpleTreeNode::add() is only allowed in device code.");
#endif
	}

	KRR_CALLABLE SimpleTreeNode operator*(const float val) const {
		SimpleTreeNode result;
		result.mVal = mVal * val;
		return result;
	}

	KRR_CALLABLE SimpleTreeNode operator/(const float val) const {
		SimpleTreeNode result;
		result.mVal = mVal / val;
		return result;
	}

	KRR_CALLABLE SimpleTreeNode operator*(const SimpleTreeNode &val) const {
		SimpleTreeNode result;
		result.mVal = mVal * val.mVal;
		return result;
	}

	KRR_CALLABLE SimpleTreeNode &operator+=(const SimpleTreeNode &rhs) {
		mVal += rhs.mVal;
		return *this;
	}

	KRR_CALLABLE SimpleTreeNode() { this->mVal = 0; }
	KRR_CALLABLE SimpleTreeNode(const Spectrum &val) { this->mVal = val.mean(); }

	KRR_DEVICE bool isinf() const { return ::isinf(mVal); }
	KRR_HOST SimpleTreeNode sqrtf() const {
		SimpleTreeNode result;
		result.mVal = ::sqrtf(mVal);
		return result;
	}
};

struct RadianceTreeNode {
	float mVal;
	Spectrum mRadiance;

	KRR_HOST void store(const float val) {
		mVal	  = val;
		mRadiance = Spectrum(val);
	}

	KRR_HOST void store(const RadianceTreeNode &node) {
		mVal	  = node.mVal;
		mRadiance = node.mRadiance;
	}

	KRR_CALLABLE float load() const { return mVal; }

	KRR_CALLABLE RadianceTreeNode loadAll() const { return *this; }

	KRR_DEVICE void add(RadianceTreeNode val) {
#ifdef KRR_DEVICE_CODE
		atomicAdd(&mVal, val.mVal);
		atomicAdd(((float *) &mRadiance) + 0, val.mRadiance.x());
		atomicAdd(((float *) &mRadiance) + 1, val.mRadiance.y());
		atomicAdd(((float *) &mRadiance) + 2, val.mRadiance.z());
#else
		Log(Fatal, "RadianceTreeNode::add() is only allowed in device code.");
#endif
	}

	KRR_CALLABLE RadianceTreeNode operator*(const float val) const {
		RadianceTreeNode result;
		result.mVal		 = mVal * val;
		result.mRadiance = mRadiance * val;
		return result;
	}

	KRR_CALLABLE RadianceTreeNode operator/(const float val) const {
		RadianceTreeNode result;
		result.mVal		 = mVal / val;
		result.mRadiance = mRadiance / val;
		return result;
	}

	KRR_CALLABLE RadianceTreeNode operator*(const RadianceTreeNode &val) const {
		RadianceTreeNode result;
		result.mVal		 = mVal * val.mVal;
		result.mRadiance = mRadiance * val.mRadiance;
		return result;
	}

	KRR_CALLABLE RadianceTreeNode &operator+=(const RadianceTreeNode &rhs) {
		mVal += rhs.mVal;
		mRadiance += rhs.mRadiance;
		return *this;
	}

	KRR_CALLABLE RadianceTreeNode() {
		this->mVal		= 0;
		this->mRadiance = Spectrum(0);
	}

	KRR_CALLABLE RadianceTreeNode(const Spectrum &val) {
		this->mVal		= val.mean();
		this->mRadiance = val;
	}

	KRR_HOST RadianceTreeNode sqrtf() const {
		RadianceTreeNode result;
		result.mVal		 = ::sqrtf(mVal);
		result.mRadiance = mRadiance.sqrt(); // OK?
		return result;
	}

	KRR_DEVICE bool isinf() const { return ::isinf(mVal) || mRadiance.isInf().any(); }
};

NAMESPACE_END(krr)