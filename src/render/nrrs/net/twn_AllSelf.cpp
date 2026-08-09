#include "twn_AllSelf.h"

NAMESPACE_BEGIN(krr)

void TwoHeadedNetwork_AllSelf::reset() {
	json encoding_opts	= m_config.value("encoding", json::object());
	json optimizer_opts = m_config.value("optimizer", json::object());

	static bool firstLoad = true;
	if (firstLoad) {
		float lr = optimizer_opts["nested"]["learning_rate"];

		optimizer_opts["nested"]["learning_rate"] = lr / 5.0f; // for stability

		firstLoad = false;
	}

	json twoNet_config = m_config.value("two_head_net_all_self", json::object());

	const int n_neurons = m_config["n_neurons"];

	json net_opts	= twoNet_config.value("net", json::object());
	json loss_opts1 = twoNet_config.value("loss1", json::object());
	json loss_opts2 = twoNet_config.value("loss2", json::object());

	net_opts["n_neurons"] = n_neurons;

	json loss_opts[2] = {loss_opts1, loss_opts2};

	for (int net_idx = 0; net_idx < 2; ++net_idx) {
		auto &net	  = m_net[net_idx];
		auto &opti	  = m_optimizer[net_idx];
		auto &trainer = m_trainer[net_idx];
		auto &loss	  = m_loss[net_idx];

		loss.reset(create_loss<precision_t>(loss_opts[net_idx]));
		net = std::make_shared<NetworkWithInputEncoding>(m_n_input_dims, m_dims_output,
														 encoding_opts, net_opts);
		opti.reset(create_optimizer<precision_t>(optimizer_opts));
		trainer = std::make_shared<Trainer_Float>(net, opti, loss);
	}
}

TwoHeadedNetwork_AllSelf::TwoHeadedNetwork_AllSelf(const json &config, const uint32_t n_input_dims,
												   const uint32_t dims_output) {
	// L:3, L2:3
	assert(dims_output == 6);

	m_n_input_dims = n_input_dims;
	m_dims_output  = dims_output / 2; // 3

	m_config = config;

	reset();
}

void TwoHeadedNetwork_AllSelf::set_max_training_batch_size(const uint32_t batch_size) { return; }

void TwoHeadedNetwork_AllSelf::set_max_inference_batch_size(const uint32_t batch_size) { return; }

void TwoHeadedNetwork_AllSelf::inference(cudaStream_t stream, const GPUMatrix_Float &input,
										 GPUMatrix_Half &output1, GPUMatrix_Half &output2) {

	m_net[0]->inference_mixed_precision(stream, input, output1);
	m_net[1]->inference_mixed_precision(stream, input, output2);
}

float TwoHeadedNetwork_AllSelf::training_step(cudaStream_t stream, const GPUMatrix_Float &input,
											  const GPUMatrix_Float &target, const bool get_loss,
											  precision_t *&output1, precision_t *&output2) {

	float ret = 0.0f;

	const float loss_scale	 = default_loss_scale<precision_t>();
	const uint32_t batchSize = input.n();

	std::unique_ptr<Trainer_Float::ForwardContext> ctxs[2];
	{
		// Execute forward and backward in a CUDA graph for maximum performance.
		auto capture_guard = m_graph.capture_guard(stream);

		CHECK_THROW(target.m() == 6);
		// just for pass the forward check, in loss.evaualte(), we decode the target
		GPUMatrix_Float target_proxy(target.data(), 3, target.n());

		// [0]->forward() also generate from net1->out => net2->out in loss.evaualte()

		for (int net_idx = 0; net_idx < 2; ++net_idx) {
			auto &ctx	  = ctxs[net_idx];
			auto &trainer = m_trainer[net_idx];

			ctx = trainer->forward(stream, loss_scale, input, target_proxy, nullptr, false, false,
								   nullptr);
			trainer->backward(stream, *ctx, input);
			trainer->optimizer_step(stream, loss_scale);
		}

		output1 = ctxs[0]->output.data();
		output2 = ctxs[1]->output.data();
	}

	// [STEP 4] calcuate the loss if needed
	if (get_loss) {
		float loss1 = m_trainer[0]->loss(stream, *ctxs[0]);
		float loss2 = m_trainer[1]->loss(stream, *ctxs[1]);

		ret = (loss1 + loss2) / 2.0f;
	}

	return ret;
}

void TwoHeadedNetwork_AllSelf::updateLossHyperparams(const json &config) {
	m_loss[0]->update_hyperparams(config);
	m_loss[1]->update_hyperparams(config);
}

void TwoHeadedNetwork_AllSelf::print_info() const {
	int net_params		 = 0;
	int encoding_params	 = 0;
	int trainer_params	 = 0;
	int optimizer_params = 0;

	for (int net_idx = 0; net_idx < 2; ++net_idx) {
		encoding_params += m_net[net_idx]->encoding()->n_params();
		net_params += m_net[net_idx]->n_params();
		trainer_params += m_trainer[net_idx]->n_params();
		optimizer_params += m_optimizer[net_idx]->n_weights();
	}

	Log(Info,
		"Network has %d (%d + %d) parameters."
		" Trainer has %d parameters."
		" Optimizer has %d parameters.",
		net_params, encoding_params, (net_params - encoding_params), trainer_params,
		optimizer_params);
}

NAMESPACE_END(krr)
