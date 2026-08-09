#include "twn_PartShare.h"
#include "../train.h"

NAMESPACE_BEGIN(krr)

void TwoHeadedNetwork_PartShare::reset() {
	json encoding_opts	= m_config.value("encoding", json::object());
	json optimizer_opts = m_config.value("optimizer", json::object());

	json twoNet_config = m_config.value("two_head_net_part_share", json::object());

	const int n_neurons = m_config["n_neurons"];

	json loss_opts1	  = twoNet_config.value("loss1", json::object());
	json loss_opts2	  = twoNet_config.value("loss2", json::object());
	json network_opts = twoNet_config.value("base_net", json::object());
	json net12_opts	  = twoNet_config.value("tail_net", json::object());
	m_dims_transfer	  = n_neurons;

	std::cout << "[TwoHeadedNetwork_PartShare] Set Internal Dimension: " << m_dims_transfer
			  << std::endl;

	network_opts["n_output_dims"] = m_dims_transfer;
	net12_opts["n_input_dims"]	  = m_dims_transfer;
	net12_opts["n_output_dims"]	  = m_dims_output;
	net12_opts["n_neurons"]		  = n_neurons;

	// base_net [no loss]
	network_opts["n_neurons"] = n_neurons;
	m_net_base = std::make_shared<NetworkWithInputEncoding>(m_n_input_dims, m_dims_transfer,
															encoding_opts, network_opts);
	m_optimizer_base.reset(create_optimizer<precision_t>(optimizer_opts));
	m_trainer_base = std::make_shared<Trainer_Float>(m_net_base, m_optimizer_base, nullptr);

	json loss_opts[2] = {loss_opts1, loss_opts2};

	for (int net_idx = 0; net_idx < 2; ++net_idx) {
		// net1 & net2
		m_net_heads[net_idx].reset(create_network<precision_t>(net12_opts));
		m_optimizer_heads[net_idx].reset(create_optimizer<precision_t>(optimizer_opts));
		m_loss_heads[net_idx].reset(create_loss<precision_t>(loss_opts[net_idx]));
		m_trainer_heads[net_idx] = std::make_shared<Trainer_Half>(
			m_net_heads[net_idx], m_optimizer_heads[net_idx], m_loss_heads[net_idx]);
	}
}

TwoHeadedNetwork_PartShare::TwoHeadedNetwork_PartShare(const json &config,
													   const uint32_t n_input_dims,
													   const uint32_t dims_output) {
	// L:3, L2:3
	assert(dims_output == 6);

	m_n_input_dims = n_input_dims;
	m_dims_output  = dims_output / 2; // 3

	m_config = config;

	reset();
}

void TwoHeadedNetwork_PartShare::set_max_training_batch_size(const uint32_t batch_size) {
	if (m_max_training_batch_size == batch_size) {
		return;
	}

	m_max_training_batch_size = batch_size;

	uint32_t size = m_dims_transfer * batch_size;

	m_transfer_buffer_dL_dy	 = GPUMemory<precision_t>(size);
	m_transfer_buffer_dL_dy2 = GPUMemory<precision_t>(size);
}

void TwoHeadedNetwork_PartShare::set_max_inference_batch_size(const uint32_t batch_size) {
	if (m_max_inference_batch_size == batch_size) {
		return;
	}
	m_max_inference_batch_size = batch_size;

	m_transfer_buffer_inference = GPUMemory<precision_t>(m_dims_transfer * batch_size);
}

void TwoHeadedNetwork_PartShare::inference(cudaStream_t stream, const GPUMatrix_Float &input,
										   GPUMatrix_Half &output1, GPUMatrix_Half &output2) {

	GPUMatrix_Half transfer_l(m_transfer_buffer_inference.data(), m_dims_transfer, input.cols());

	m_net_base->inference_mixed_precision(stream, input, transfer_l);
	m_net_heads[0]->inference_mixed_precision(stream, transfer_l, output1);
	m_net_heads[1]->inference_mixed_precision(stream, transfer_l, output2);
}

float TwoHeadedNetwork_PartShare::training_step(cudaStream_t stream, const GPUMatrix_Float &input,
												const GPUMatrix_Float &target, const bool get_loss,
												precision_t *&output1, precision_t *&output2) {

	float ret = 0.0f;

	const float loss_scale	 = default_loss_scale<precision_t>();
	const uint32_t batchSize = input.n();

	std::unique_ptr<Trainer_Float::ForwardContext> ctx_base;
	std::unique_ptr<Trainer_Half::ForwardContext> ctx1, ctx2;
	{
		// Execute forward and backward in a CUDA graph for maximum performance.
		auto capture_guard = m_graph.capture_guard(stream);

		// [STEP 1] forward pass

		// [STEP 1.1] net_base forward
		// [notes]
		// (1) now we use external dL_dy, so we don't use target in forward()
		// (2) here only set m_transfer_buffer_dL_dy to ctx_base->dL_doutput, no computation
		static GPUMatrix_Float target_tmp;

		GPUMatrix_Half transfer_dL_dy(m_transfer_buffer_dL_dy.data(), m_dims_transfer, batchSize);
		GPUMatrix_Half transfer_dL_dy2(m_transfer_buffer_dL_dy2.data(), m_dims_transfer, batchSize);

		ctx_base = m_trainer_base->forward(stream, loss_scale, input, target_tmp, nullptr, false,
										   false, &transfer_dL_dy);

		CHECK_THROW(target.m() == 6);

		// just for pass the forward check, in loss.evaualte(), we decode the target
		GPUMatrix_Float target_proxy(target.data(), 3, target.n());

		// [STEP 1.2] net1 & net2 forward
		ctx1	= m_trainer_heads[0]->forward(stream, loss_scale, ctx_base->output, target_proxy,
											  nullptr, false, false, nullptr);
		output1 = ctx1->output.data();

		// here we should generate from net1->out => net2->out, for loss
		// we do this in the net1's loss.evaulate() in forward()

		ctx2	= m_trainer_heads[1]->forward(stream, loss_scale, ctx_base->output, target_proxy,
											  nullptr, false, false, nullptr);
		output2 = ctx2->output.data();

		// [STEP 2] backward pass
		// [STEP 2.1] net1 & net2 backward
		m_trainer_heads[0]->backward(stream, *ctx1, ctx_base->output, &transfer_dL_dy, false,
									 GradientMode::Overwrite);

		m_trainer_heads[1]->backward(stream, *ctx2, ctx_base->output, &transfer_dL_dy2, false,
									 GradientMode::Overwrite);

		// [STEP 2.2] accumulate the gradients
		linear_kernel(calculate_gradient_sum, 0, stream, m_dims_transfer * batchSize,
					  m_transfer_buffer_dL_dy.data(), m_transfer_buffer_dL_dy2.data());

		// [STEP 2.3] net_base backward
		m_trainer_base->backward(stream, *ctx_base, input, nullptr, false, GradientMode::Overwrite);

		// [STEP 3] optimizer step
		m_trainer_heads[0]->optimizer_step(stream, loss_scale);
		m_trainer_heads[1]->optimizer_step(stream, loss_scale);
		m_trainer_base->optimizer_step(stream, loss_scale);
	}

	// [STEP 4] calcuate the loss if needed
	if (get_loss) {
		float loss1 = m_trainer_heads[0]->loss(stream, *ctx1);
		float loss2 = m_trainer_heads[1]->loss(stream, *ctx2);

		ret = (loss1 + loss2) / 2.0f;
	}

	return ret;
}

void TwoHeadedNetwork_PartShare::updateLossHyperparams(const json &config) {
	m_loss_heads[0]->update_hyperparams(config);
	m_loss_heads[1]->update_hyperparams(config);
}

void TwoHeadedNetwork_PartShare::print_info() const {
	int net_params = 0;
	net_params += m_net_base->n_params();
	net_params += m_net_heads[0]->n_params();
	net_params += m_net_heads[1]->n_params();

	int encoding_params = m_net_base->encoding()->n_params();

	int trainer_params = 0;
	trainer_params += m_trainer_base->n_params();
	trainer_params += m_trainer_heads[0]->n_params();
	trainer_params += m_trainer_heads[1]->n_params();

	int optimizer_params = 0;
	optimizer_params += m_optimizer_base->n_weights();
	optimizer_params += m_optimizer_heads[0]->n_weights();
	optimizer_params += m_optimizer_heads[1]->n_weights();

	Log(Info,
		"Network has %d (%d + %d) parameters."
		" Trainer has %d parameters."
		" Optimizer has %d parameters.",
		net_params, encoding_params, (net_params - encoding_params), trainer_params,
		optimizer_params);
}

NAMESPACE_END(krr)
