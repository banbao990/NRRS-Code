#pragma once
#include "common.h"

// 0: input = spatial + directional
// 1: input = spatial + directional + normal + roughness
#define NETWORK_AUXILIARY_INPUT 0

constexpr unsigned int NRC_MAX_TRAIN_DEPTH		 = 4; // max training samples per pixel
constexpr unsigned int NRC_DIM_SPATIAL_INPUT	 = 3; // spatial data (positions)
constexpr unsigned int NRC_DIM_DIRECTIONAL_INPUT = 2; // MLP(x, w_i) -> L (w_i)

// how many dims do auxiliary data (e.g., normals, roughness) have?
constexpr unsigned int NRC_DIM_AUXILIARY_INPUT = NETWORK_AUXILIARY_INPUT ? 3 : 0;

// how many dims do the network input [spatial + auxiliary] have?
constexpr unsigned int NRC_DIM_INPUT =
	NRC_DIM_SPATIAL_INPUT + NRC_DIM_DIRECTIONAL_INPUT + NRC_DIM_AUXILIARY_INPUT;
constexpr unsigned int NRC_DIM_OUTPUT = 3; // only rgb

// max size of the rendering frame
// TODO: make it dynamic
constexpr unsigned int NRC_MAX_RESOLUTION = 1280 * 720;
// [resolution-affected]
constexpr int NRC_TRAIN_BUFFER_SIZE	  = NRC_MAX_TRAIN_DEPTH * NRC_MAX_RESOLUTION;
constexpr size_t NRC_TRAIN_BATCH_SIZE = 65'536 * 8;

// the minimum batch size we can tolerate (to avoid unstable training)
constexpr size_t NRC_MIN_TRAIN_BATCH_SIZE = 65'536;
constexpr int NRC_MAX_INFERENCE_NUM		  = NRC_MAX_RESOLUTION; // [resolution-affected]