#include "denoisetask.h"
#include "traditional.h"

#include <logger.h>
#include <window.h>

#include <render/profiler/profiler.h>
#include <renderpass.h>

NAMESPACE_BEGIN(krr)
extern "C" char DENOISE_TASK_PTX[];

const char *DenoiseTask::sGBufferNames[] = {"White", "Albedo", "Normal", "Color", "Denoised"};

DenoiseTask::DenoiseTask() {
	for (int i = 0; i < BufferType::Count; ++i) {
		mBuffers[i] = nullptr;
	}
}

DenoiseTask::~DenoiseTask() {
	for (int i = 0; i < BufferType::Count; ++i) {
		const RGB *buffer = mBuffers[i];
		if (buffer) {
			cudaFree((void *) buffer);
		}
	}
}

void DenoiseTask::initialize(const uint width, const uint height) {
	Log(Info, "Initialize/Reset DenoiseTask.");

	const uint pixels = width * height;
	mWidth			  = width;
	mHeight			  = height;

	if (pixels == mResolutionSize) {
		return;
	}

	const auto stream	   = gpContext->cudaStream;
	mGBufferInvalid		   = true;
	mDenoisedResultIsReady = false;
	for (int i = 0; i < BufferType::Count; ++i) {
		const RGB *buffer = mBuffers[i];
		if (buffer) {
			cudaFreeAsync((void *) buffer, stream);
		}
		cudaMallocAsync(&mBuffers[i], sizeof(RGB) * pixels, stream);
	}
	cudaStreamSynchronize(stream);
	mResolutionSize = pixels;

	{
		// as the renderer design, pass->initialize() only called once out side itself
		static bool sDenoisePassInitialized = false;
		if (!sDenoisePassInitialized) {
			mDenoisePass->initialize();
			sDenoisePassInitialized = true;
		}
	}
}

void DenoiseTask::resize(const Vector2i size) {
	initialize(size[0], size[1]);
	mDenoisePass->resize(size);
}

RGB *DenoiseTask::getBuffer(BufferType bufferType) {
	if (bufferType >= BufferType::Count) {
		Log(Error, "Invalid buffer type.");
		return nullptr;
	}
	return mBuffers[static_cast<uint>(bufferType)];
}

void DenoiseTask::resetPixels() {
	RGB *albedo = getBuffer(BufferType::Albedo);
	RGB *normal = getBuffer(BufferType::Normal);
	GPUParallelFor(
		mResolutionSize,
		[=] KRR_DEVICE(const int tid) {
			albedo[tid] = 0;
			normal[tid] = 0;
		},
		gpContext->cudaStream);
}

void DenoiseTask::setShouldUpdateColorBuffer(bool shouldUpdate) {
	mShouldUpdateColorBuffer = shouldUpdate;
}

void DenoiseTask::updateColorBuffer(Film *film) {
	if (!mShouldUpdateColorBuffer) {
		return;
	}

	RGB *color = getBuffer(BufferType::Color);
	GPUParallelFor(
		mResolutionSize,
		[=] KRR_DEVICE(const int pixelId) mutable {
			RGBA L		   = film->getPixel(pixelId);
			color[pixelId] = RGB(L);
		},
		gpContext->cudaStream);

	mShouldUpdateColorBuffer = false;
}

RGB *DenoiseTask::getSelectedBuffer() { return mBuffers[static_cast<uint>(mSelectBufferType)]; }

void DenoiseTask::renderUI(bool cannotShowBuffer) {
	ui::Checkbox("Enable Denoise", &mEnable);
	if (!mEnable) {
		return;
	}

	if (ui::Checkbox("Denoise Always", &mDenoiseAlways)) {
		// this setting will get the denoise result back(submit this pass in render())
		mDenoiseOnce = true;
	}
	if (!mDenoiseAlways) {
		ui::Checkbox("Denoise Once", &mDenoiseOnce);
	}

	bool tGBufferChanged = false;
	tGBufferChanged |= ui::Checkbox("GBuffer 1st Non-Specular", &mGBufferEnableSpecularContinue);
	tGBufferChanged |= ui::SliderInt("GBuffer Spp", &mGBufferSpp, 1, 32, "%d");
	tGBufferChanged |= ui::Checkbox("Don't Need Albedo", &mDontNeedAlbedo);
	if (mDontNeedAlbedo) {
		tGBufferChanged |=
			ui::SliderFloat("Albedo Value", &mDontNeedAlbedoValue, 0.0f, 1.0f, "%.2f");
	}
	mGBufferInvalid |= tGBufferChanged;
	mDenoiseOnce |= tGBufferChanged;

	if (cannotShowBuffer) {
		mShowGBuffer = false;
	} else {
		ui::Checkbox("Show GBuffer", &mShowGBuffer);
		if (mShowGBuffer) {
			ui::Combo("GBuffer", (int *) &mSelectBufferType, sGBufferNames,
					  sizeof(sGBufferNames) / sizeof(char *));
		}
	}

	if (ui::TreeNodeEx("Denoise Pass", ImGuiTreeNodeFlags_DefaultOpen)) {
		ui::PushID("DenoisePass");
		mDenoisePass->renderUI();

		ui::PopID();
		ui::TreePop();
	}
}

void DenoiseTask::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	if (!mBackend) {
		mBackend	= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(DENOISE_TASK_PTX)
						  .addRaygenEntry("GBuffer")
						  .addRayType("GBuffer", true, true, false);

		mBackend->initialize(params);
	}
	mBackend->setScene(scene);
}

void DenoiseTask::renderGBuffer() {
	if (!mEnable || !mGBufferInvalid || !mDenoisePass->useGeometry()) {
		return;
	}

	++mDenoiseFrameId;

	PROFILE("Render GBuffer");

	// reset buffers
	resetPixels();

	// MegaKernel Style
	static LaunchParameters<DenoiseTask> params = {};

	params.mCamera						  = mScene->getCamera()->getCameraData();
	params.mSceneData					  = mBackend->getSceneData();
	params.mTraversable					  = mBackend->getRootTraversable();
	params.mFrameBufferSize				  = Vector2i(mWidth, mHeight);
	params.mGBufferSpp					  = mGBufferSpp;
	params.mDenoiseFrameId				  = mDenoiseFrameId;
	params.mAlbedoBuffer				  = getBuffer(DenoiseTask::Albedo);
	params.mNormalBuffer				  = getBuffer(DenoiseTask::Normal);
	params.mGBufferEnableSpecularContinue = mGBufferEnableSpecularContinue;
	params.mGBufferSppInv				  = 1.0f / mGBufferSpp;

	mBackend->launch(params, "GBuffer", mWidth, mHeight, 1);

	// don't need albedo
	{
		RGB *albedo			  = getBuffer(BufferType::White);
		const float albedoVar = mDontNeedAlbedoValue;
		GPUParallelFor(
			mResolutionSize, [=] KRR_DEVICE(const int tid) { albedo[tid] = albedoVar; },
			gpContext->cudaStream);
	}

	mGBufferInvalid = false;
	Log(Info, "Render GBuffer Done.");
}

bool DenoiseTask::shouldDenoise() const { return mDenoiseOnce || mDenoiseAlways; }

void DenoiseTask::writeGBuffer(CudaRenderTarget &frameBuffer, uint32_t scale) {
	if (!mEnable || !mShowGBuffer) {
		return;
	}

	const RGB *buffer		 = getSelectedBuffer();
	auto stream				 = gpContext->cudaStream;
	const int resolutionSize = mResolutionSize;

	if (scale != 1) {
		const int sizeDesired  = frameBuffer.width * frameBuffer.height;
		const int widthDenoise = mWidth;
		// should scale
		GPUParallelFor(
			sizeDesired,
			[=] KRR_DEVICE(const int pixelId) mutable {
				if (pixelId >= sizeDesired) {
					return;
				}

				int ox = pixelId % frameBuffer.width;
				int oy = pixelId / frameBuffer.width;
				ox /= scale;
				oy /= scale;

				int pixelIdScaled = ox + oy * widthDenoise;

				const RGB L = buffer[pixelIdScaled];
				frameBuffer.write(RGBA(L, 1), pixelId);
			},
			stream);
	} else {
		GPUParallelFor(
			mResolutionSize,
			[=] KRR_DEVICE(const int pixelId) mutable {
				if (pixelId >= resolutionSize) {
					return;
				}
				const RGB L = buffer[pixelId];
				frameBuffer.write(RGBA(L, 1), pixelId);
			},
			stream);
	}
}

void DenoiseTask::resetState() {
	mDenoisedResultIsReady = false;
	mDenoiseOnce		   = true;
}

void DenoiseTask::denoise(const bool log, RGB *resultBuffer) {
	bool tShouldDenoise = mEnable && shouldDenoise();

	if (tShouldDenoise) {
		PROFILE("DenoiseTask -> Denoise");
		mDenoisePass->setEnable(true);

		RGB *albedo =
			mDontNeedAlbedo ? getBuffer(DenoiseTask::White) : getBuffer(DenoiseTask::Albedo);
		RGB *result = resultBuffer ? resultBuffer : getBuffer(DenoiseTask::DenoisedColor);

		mDenoisePass->denoise((float *) getBuffer(DenoiseTask::Color), (float *) result,
							  DenoiseBackend::PixelFormat::FLOAT3,
							  (float *) getBuffer(DenoiseTask::Normal), (float *) albedo);

		// auto stream = gpContext->cudaStream;

		// LinearKernel(GaussianFilter, stream, mResolutionSize, 5, mWidth, mHeight,
		//			 (float *) getBuffer(DenoiseTask::Color),
		//			 (float *) getBuffer(DenoiseTask::DenoisedColor));

		// LinearKernel(BilateralFilter, stream, mResolutionSize, 5, mWidth, mHeight,
		//			 (float *) getBuffer(DenoiseTask::Color),
		//			 (float *) getBuffer(DenoiseTask::Normal),
		//			 (float *) getBuffer(DenoiseTask::DenoisedColor));

		mDenoisedResultIsReady = true;
		mDenoiseOnce		   = false;
		// Log(Success, "Denoised Successfully");
	}
}

NAMESPACE_END(krr)