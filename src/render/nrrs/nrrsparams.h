#pragma once

constexpr size_t NRRS_TRAIN_BATCH_SIZE = 65'536 * 8;

// the minimum batch size we can tolerate (to avoid unstable training)
constexpr size_t NRRS_MIN_TRAIN_BATCH_SIZE = 65'536;

constexpr size_t NRRS_MAX_TRAIN_DEPTH = 4;

// will **always** use camera direction as training input
// #define NRRS_USE_CAMERA_DIRECTION

#define NRRS_USE_ROUGHNESS

#define NRRS_USE_REF_MEAN

// [TODO] should carefully use this, now we do not check this macro
// #define NRRS_USE_POS_DIR
