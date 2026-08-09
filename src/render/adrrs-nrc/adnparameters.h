#pragma once

// as we use the structs from the nrc code, we should have the same MACROs
#include "render/nrc/nrcparameters.h"

// use camera/ray direction as training input
#define USE_CAMERA_DIRECTION

// use ui to control the parameters(better for debug)
// constexpr float RRS_CLAMP_MAX = 5.0f, RRS_CLAMP_MIN = 0.5f;