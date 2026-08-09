#pragma once
#include <common.h>
#include <render/color.h>
#include <device/cuda.h>
#include <util/film.h>

#include <render/passes/denoise/denoise.h>

NAMESPACE_BEGIN(krr)

// Megakernel Style
class DenoiseTask {
public:
	enum BufferType : uint {
		White, // (v,v,v) according to mDontNeedAlbedoValue
		Albedo,
		Normal,
		Color,
		DenoisedColor,
		Count
	};

public:
	DenoiseTask();
	~DenoiseTask();

	bool isEnabled() const { return mEnable; }
	void setEnabled(bool enable) { mEnable = enable; }
	void initialize(const uint width, const uint height);
	void resize(const Vector2i size);
	RGB *getBuffer(BufferType bufferType);
	void resetPixels();
	void setShouldUpdateColorBuffer(bool shouldUpdate);
	void updateColorBuffer(Film *film);
	RGB *getSelectedBuffer();
	void renderUI(bool cannotShowBuffer = false);

	void setScene(Scene::SharedPtr scene);
	void renderGBuffer();
	bool shouldDenoise() const;
	void writeGBuffer(CudaRenderTarget &frameBuffer, uint32_t scale = 1u);
	bool isDenoisedResultReady() const { return mDenoisedResultIsReady; }
	void resetState();

	void setGBufferInvalid() { mGBufferInvalid = true; }
	void setDontNeedAlbedo(bool dontNeedAlbedo) { mDontNeedAlbedo = dontNeedAlbedo; }
	bool getDontNeedAlbedo() const { return mDontNeedAlbedo; }

	// will check whether denoising is needed in the denoise() function
	void denoise(const bool log, RGB *resultBuffer = nullptr);

	void from_json(const json &j) {
		mEnable = j.value("enable", true);
		// see: renderer.cpp
		mDenoisePass  = std::make_shared<DenoisePass>();
		auto params	  = j.value("denoisepass_params", json{});
		*mDenoisePass = params.get<DenoisePass>();
	}

public:
	const static char *sGBufferNames[];

private:
	bool mEnable{true};

	RGB *mBuffers[BufferType::Count] = {}; // all buffers
	// the number of pixels in the resolution
	uint mResolutionSize{0};
	uint mWidth{0};
	uint mHeight{0};
	// denoise one frame
	bool mDenoiseOnce{false};
	// denoise every frame
	bool mDenoiseAlways{false};
	// true: find the first non-specular hit point recursively
	bool mGBufferEnableSpecularContinue{true};
	// the number of samples per pixel to renderGBuffer()
	int mGBufferSpp{16};
	// the number of times renderGBuffer() has been called, also random seed for renderGBuffer()
	uint mDenoiseFrameId{0};
	// should write the GBuffer to frameBuffer
	bool mShowGBuffer{false};
	// true: the denoised result is ready
	bool mDenoisedResultIsReady{false};
	// true: all buffers are allocated in GPU
	bool mInitialized{false};
	// true: get the color buffer from the Film when Film is updated
	bool mShouldUpdateColorBuffer{false};
	// the select buffer to show
	BufferType mSelectBufferType{BufferType::Albedo};
	// true: don't need the albedo buffer; set all to mDontNeedAlbedoValue
	bool mDontNeedAlbedo{false};
	float mDontNeedAlbedoValue{1.0f};

	bool mGBufferInvalid{true};

	std::shared_ptr<DenoisePass> mDenoisePass;
	OptixBackend *mBackend{nullptr};
	Scene::SharedPtr mScene{nullptr};
};

template <> struct LaunchParameters<DenoiseTask> {
	rt::CameraData mCamera;
	rt::SceneData mSceneData;
	OptixTraversableHandle mTraversable;
	Vector2i mFrameBufferSize;

	uint mGBufferSpp;
	uint mDenoiseFrameId;
	RGB *mAlbedoBuffer;
	RGB *mNormalBuffer;
	bool mGBufferEnableSpecularContinue;
	float mGBufferSppInv;
};
NAMESPACE_END(krr)