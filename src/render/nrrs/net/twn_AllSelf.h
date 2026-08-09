#pragma once

#include "twoHeadedNetwork.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

class TwoHeadedNetwork_AllSelf : public TwoHeadedNetwork {
public:
	TwoHeadedNetwork_AllSelf(const TwoHeadedNetwork_AllSelf &)			  = delete;
	TwoHeadedNetwork_AllSelf &operator=(const TwoHeadedNetwork_AllSelf &) = delete;

	TwoHeadedNetwork_AllSelf(const json &config, const uint32_t n_input_dims,
							 const uint32_t dims_output);
	~TwoHeadedNetwork_AllSelf() override {}

	void reset() override;
	void print_info() const override;

	void set_max_training_batch_size(const uint32_t batch_size) override;
	void set_max_inference_batch_size(const uint32_t batch_size) override;

	void inference(cudaStream_t stream, const GPUMatrix_Float &input, GPUMatrix_Half &output1,
				   GPUMatrix_Half &output2) override;

	float training_step(cudaStream_t stream, const GPUMatrix_Float &input,
						const GPUMatrix_Float &target, const bool get_loss, precision_t *&output1,
						precision_t *&output2) override;

	void updateLossHyperparams(const json &config) override;

private:
	CudaGraph m_graph;
	json m_config;

	uint32_t m_n_input_dims;
	uint32_t m_dims_output;

	// net1 & net2
	std::shared_ptr<NetworkWithInputEncoding> m_net[2];
	std::shared_ptr<Optimizer> m_optimizer[2];
	std::shared_ptr<Trainer_Float> m_trainer[2];
	std::shared_ptr<Loss> m_loss[2];
};

NAMESPACE_END(krr)