#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"

#include "util/film.h"

NAMESPACE_BEGIN(krr)

class LoadImagePass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<LoadImagePass>;
	KRR_REGISTER_PASS_DEC(LoadImagePass);

	LoadImagePass() = default;
	void finalize() override {
		auto &alloc = gpContext->alloc;
		alloc->deallocate_object(mImage, mImageSize.x() * mImageSize.y());
	}

	void renderUI() override {
		ui::Checkbox("Enabled", &mEnable);
		if (!mEnable) return;

		if (ui::InputText("Image Path", mImagePathBuffer, sizeof(mImagePathBuffer))) {
			mImagePath = mImagePathBuffer;
		}
		if (ui::Button("Load")) {
			loadImage();
		}
	}

	void render(RenderContext *context) override {
		if (mImage) {
			Vector2i frameSize = getFrameSize();
			Vector2i maxSize;
			maxSize[0] = min(mImageSize[0], frameSize[0]);
			maxSize[1] = min(mImageSize[1], frameSize[1]);

			CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();
			GPUParallelFor(
				maxSize.x() * maxSize.y(),
				KRR_DEVICE_LAMBDA(const int tid) {
					int x = tid % maxSize.x();
					int y = tid / maxSize.x();
					if (x < mImageSize.x() && y < mImageSize.y()) {
						RGBA L		= mImage->getPixel(Vector2i(x, y));
						int pixelId = y * frameSize.x() + x;
						frameBuffer.write(L, pixelId);
					}
				},
				gpContext->cudaStream);
		}
	}

	void loadImage() {
		auto &alloc		  = gpContext->alloc;
		const auto stream = gpContext->cudaStream;

		auto img	 = std::make_shared<Image>();
		bool success = img->loadImage(mImagePath, true, false);
		if (success) {
			// TODO: find out why saving an exr image yields this permutation on pixel format?
			// This should be deleted once new reference images are updated.
			auto permute = [](auto pixel) {
				Array4i p = {3, 0, 1, 2};
				auto res  = pixel;
				for (int c = 0; c < 4; c++) res[c] = pixel[p[c]];
				return res;
			};
			img->process(permute);
			RGBA *data;
			cudaMalloc(&data, sizeof(RGBA) * img->getSizeInBytes() / sizeof(RGBA));
			cudaMemcpy(data, reinterpret_cast<RGBA *>(img->data()),
					   sizeof(RGBA) * img->getSizeInBytes() / sizeof(RGBA), cudaMemcpyHostToDevice);

			cudaStreamSynchronize(stream);
			mImageSize = img->getSize();
			if (!mImage) {
				mImage = alloc->new_object<Film>(mImageSize);
			} else {
				mImage->resize(mImageSize);
			}

			GPUParallelFor(
				mImageSize.x() * mImageSize.y(),
				KRR_DEVICE_LAMBDA(const int pixelId) { mImage->put(data[pixelId], pixelId); },
				stream);
			cudaFreeAsync(data, stream);

			Log(Success, "[LoadImegePass] Loaded image from %s.", mImagePath.c_str());
			strcpy(mImagePathBuffer, mImagePath.c_str());
		} else {
			Log(Error, "[LoadImegePass] Failed to load image from %s", mImagePath.c_str());
		}
	}

	string getName() const override { return "LoadImagePass"; }

private:
	friend void to_json(json &j, const LoadImagePass &p) {
		j = json{{"image_path", p.mImagePath}, {"is_exr", p.mIsExr}};
	}

	friend void from_json(const json &j, LoadImagePass &p) {
		p.mIsExr	 = j.value("is_exr", true);
		p.mImagePath = j.value("image_path", "");

		p.loadImage();
	}

	Film *mImage{nullptr};
	std::string mImagePath;
	char mImagePathBuffer[256]{};
	bool mIsExr{false};
	Vector2i mImageSize{0, 0};
};

KRR_REGISTER_PASS_DEF(LoadImagePass);

NAMESPACE_END(krr)