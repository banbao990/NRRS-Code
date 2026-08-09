#pragma once

#include <common.h>
#include <window.h>
#include <file.h>

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/cuda_graph.h>

namespace tcnn {
template <typename T> class Loss;
template <typename T> class Optimizer;
template <typename T> class Encoding;
template <typename T> class GPUMemory;
template <typename T> class GPUMatrixDynamic;
template <typename T, typename PARAMS_T> class Network;
template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
template <uint32_t N_DIMS, uint32_t RANK, typename T> class TrainableBuffer;
} // namespace tcnn

NAMESPACE_BEGIN(krr)

using nlohmann::json;
using precision_t			   = tcnn::network_precision_t;
using Network_Float			   = tcnn::Network<float, precision_t>;
using Network_Half			   = tcnn::Network<precision_t, precision_t>;
using Network_LL2			   = Network_Float;
using Network_RRS			   = Network_Half;
using Optimizer				   = tcnn::Optimizer<precision_t>;
using Loss					   = tcnn::Loss<precision_t>;
using Trainer_Float			   = tcnn::Trainer<float, precision_t, precision_t>;
using Trainer_Half			   = tcnn::Trainer<precision_t, precision_t, precision_t>;
using Trainer_LL2			   = Trainer_Float;
using Trainer_RRS			   = Trainer_Half;
using Trainer_RRS_AID		   = Trainer_Float;
using Encoding				   = tcnn::Encoding<precision_t>;
using NetworkWithInputEncoding = tcnn::NetworkWithInputEncoding<precision_t>;
using GPUMatrix_Float		   = tcnn::GPUMatrix<float>;
using GPUMatrix_Half		   = tcnn::GPUMatrix<precision_t>;

using Context_Float = std::shared_ptr<Trainer_Float::ForwardContext>;
using Context_Half	= std::shared_ptr<Trainer_Half::ForwardContext>;

class NRRSInferenceQueue;
class Film;
struct NRRSNetworkInput1;
struct NRRSNetworkOutputTraining1;
class MyNetworkLL2;

KRR_DEVICE_FUNCTION float activationSigma(float x) {
	// softplus
	// relu

	// activation: softplus & y = 0.5x + ln2
	constexpr float ln2 = 0.6931471805599453f;
	return (x < 0) ? (logf(1.0f + expf(x))) : (0.5f * x + ln2);
}

NAMESPACE_END(krr)
