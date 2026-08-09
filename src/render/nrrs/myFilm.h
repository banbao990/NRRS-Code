#pragma once
#include <util/film.h>

NAMESPACE_BEGIN(krr)

KRR_CALLABLE uint32_t offsetFrame2Scaled(const uint32_t offset, const uint32_t frameSizeWidth,
										 const uint32_t scale) {
	// offset is from size `mSize`, we should transform it to `mSizeScaled`
	uint32_t ox			  = offset % frameSizeWidth;
	uint32_t oy			  = offset / frameSizeWidth;
	uint32_t sx			  = ox / scale;
	uint32_t sy			  = oy / scale;
	uint32_t offsetScaled = sx + sy * (frameSizeWidth / scale);
	return offsetScaled;
}

class MyFilm {
public:
	using Pixel			= Film::Pixel;
	using WeightedPixel = Film::WeightedPixel;

	MyFilm()  = default;
	~MyFilm() = default;

	KRR_HOST MyFilm(const Vector2i frameSize, const uint32_t scale, const Vector2i size) :
		mFilm(size), mSize(frameSize), mSizeScaled(size), mScale(scale) {

		if (frameSize[0] / scale != size[0] || frameSize[1] / scale != size[1]) {
			Log(Fatal, "Frame size and scaled size do not match!");
		}
	}

	KRR_HOST void reset(const Pixel &value = {}) { mFilm.reset(value); }
	KRR_HOST void resize(const Vector2i &size) { mFilm.resize(size); }

	KRR_CALLABLE void put(const Pixel &pixel, const uint32_t offset) {
		int offsetScaled = offsetFrame2Scaled(offset, uint(mSize[0]), mScale);
		// should use atomic operation
		WeightedPixel *data = mFilm.data() + offsetScaled;
#ifdef __CUDA_ARCH__
		atomicAdd(&(data->pixel[0]), pixel[0]);
		atomicAdd(&(data->pixel[1]), pixel[1]);
		atomicAdd(&(data->pixel[2]), pixel[2]);
		atomicAdd(&(data->pixel[3]), pixel[3]);
		atomicAdd(&(data->weight), 1.f);
#else
		// we don't use this function in CPU code
		// so we don't need to use atomic operation
		Log(Fatal, "put() should not be called in CPU code!");
#endif
	};

	KRR_CALLABLE Pixel getPixelNoTransform(const size_t offset) { return mFilm.getPixel(offset); }

	KRR_CALLABLE Pixel getPixel(const size_t offset) {
		int offsetScaled = offsetFrame2Scaled(offset, uint(mSize[0]), mScale);
		return mFilm.getPixel(offsetScaled);
	}

	KRR_CALLABLE Vector2i size() { return mFilm.size(); }

private:
	Film mFilm;
	Vector2i mSize;		  // frame size
	Vector2i mSizeScaled; // real size
	uint32_t mScale{1};
};

NAMESPACE_END(krr)