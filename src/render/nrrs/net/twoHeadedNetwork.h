#pragma once

#include "networkcommon.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

class TwoHeadedNetwork {
public:
	TwoHeadedNetwork()						   = default;
	TwoHeadedNetwork(const TwoHeadedNetwork &) = delete;

	TwoHeadedNetwork(const json &config, const uint32_t n_input_dims, const uint32_t dims_output);
	TwoHeadedNetwork &operator=(const TwoHeadedNetwork &) = delete;
	virtual ~TwoHeadedNetwork() {}

	virtual void reset()			= 0;
	virtual void print_info() const = 0;

	virtual void set_max_training_batch_size(const uint32_t batch_size)	 = 0;
	virtual void set_max_inference_batch_size(const uint32_t batch_size) = 0;

	virtual void inference(cudaStream_t stream, const GPUMatrix_Float &input,
						   GPUMatrix_Half &output1, GPUMatrix_Half &output2) = 0;

	virtual float training_step(cudaStream_t stream, const GPUMatrix_Float &input,
								const GPUMatrix_Float &target, const bool get_loss,
								precision_t *&output1, precision_t *&output2) = 0;

	virtual void updateLossHyperparams(const json &config) = 0;
};

NAMESPACE_END(krr)