#pragma once

#include "device/optix.h"
#include "render/bsdf.h"
#include "render/lightsampler.h"
#include "render/wavefront/workqueue.h"
#include "workqueue.h"
#include "sampler.h"
#include "scene.h"
#include "window.h"

NAMESPACE_BEGIN(krr)

class SWPTPixelStateBuffer;
class NRCPathPixelStateBuffer;

class NRCTrainState {
public:
	KRR_CALLABLE bool isTrainingPixel(uint pixelId) const {
		return (pixelId - trainPixelOffset) % trainPixelStride == 0;
	}

	KRR_HOST void renderUI() { ImGui::InputInt("Train Pixel Stride", (int *) &trainPixelStride); }

	uint trainPixelOffset{0};
	uint trainPixelStride{1};
};

NAMESPACE_END(krr)