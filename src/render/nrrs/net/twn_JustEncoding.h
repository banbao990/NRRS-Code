#pragma once

#include "twoHeadedNetwork.h"

NAMESPACE_BEGIN(krr)
using namespace tcnn;

class TwoHeadedNetwork_JustEncoding : public TwoHeadedNetwork {
public:
	TwoHeadedNetwork_JustEncoding(const TwoHeadedNetwork_JustEncoding &)			= delete;
	TwoHeadedNetwork_JustEncoding &operator=(const TwoHeadedNetwork_JustEncoding &) = delete;

	TwoHeadedNetwork_JustEncoding(const json &config, const uint32_t n_input_dims,
								  const uint32_t dims_output);
	~TwoHeadedNetwork_JustEncoding() override {}

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
	uint32_t m_dims_transfer;
	uint32_t m_dims_output;

	// net_base
	std::shared_ptr<Encoding> m_encoding;
	std::shared_ptr<Optimizer> m_optimizer_encoding;
	std::shared_ptr<Trainer_Float> m_trainer_base;
	// net1 & net2
	std::shared_ptr<Network_Half> m_net_heads[2];
	std::shared_ptr<Optimizer> m_optimizer_heads[2];
	std::shared_ptr<Trainer_Half> m_trainer_heads[2];
	std::shared_ptr<Loss> m_loss_heads[2]; // can share the same one

	uint32_t m_max_training_batch_size{0};
	GPUMemory<precision_t> m_transfer_buffer_dL_dy{};
	GPUMemory<precision_t> m_transfer_buffer_dL_dy2{};

	uint32_t m_max_inference_batch_size{0};
	GPUMemory<precision_t> m_transfer_buffer_inference{};
};

NAMESPACE_END(krr)