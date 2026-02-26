import torch
import torch.nn as nn
from torch.autograd import Variable
from src.neuronmodel import mem_update_adp, noisy_mem_update_adp
from src import quantizationf
import logging
import numpy as np
import src.utils as utils
from functools import partial

class mlpsnn(nn.Module):
	def __init__(self, n_inputs, hidden_dim, n_outputs, decay_neu, device, seed):
		super(mlpsnn, self).__init__()

		self.device = device
		self.seed = seed
		torch.manual_seed(seed)

		self.n_inputs = n_inputs
		self.hidden_size = hidden_dim
		self.output_size = n_outputs

		self.decay_neu = decay_neu

		# nn is from torch library
		self.i2h = nn.Linear(self.n_inputs, self.hidden_size, bias=False)
		self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=False)

		self.thr_i = nn.Parameter(torch.Tensor(self.n_inputs))  # , requires_grad=True) #learn threshold
		self.thr_h = nn.Parameter(torch.Tensor(self.hidden_size))  # , requires_grad=True) #learn threshold
		self.thr_o = nn.Parameter(torch.Tensor(self.output_size))  # , requires_grad=True) #learn threshold

		nn.init.xavier_uniform_(self.i2h.weight)
		nn.init.xavier_uniform_(self.h2o.weight)

		nn.init.constant_(self.thr_h, 1.0)
		nn.init.constant_(self.thr_o, 1.0)
		nn.init.constant_(self.thr_i, 1.0)

	def forward(self, input, labels=None):
		# init
		input_mem = torch.zeros(self.n_inputs)
		hidden_mem =  torch.zeros(self.hidden_size)
		output_mem =  torch.zeros(self.output_size)
		hidden_spike = torch.zeros(self.hidden_size)
		output_spike = torch.zeros(self.output_size)

		# Feed in the whole sequence
		batch_size, seq_num, input_dimx, input_dimy = input.shape

		# Keep track of the spike train of each layer
		input_spike_train = torch.zeros((input_dimy, batch_size * self.n_inputs))
		hidden_spike_train = torch.zeros((input_dimy, batch_size * self.hidden_size))
		output_spike_train = torch.zeros((input_dimy, batch_size * self.output_size))

		# Keep track of the membrane voltages of each layer
		input_mem_train = torch.zeros((input_dimy, batch_size * self.n_inputs))
		hidden_mem_train = torch.zeros((input_dimy, batch_size * self.hidden_size))
		output_mem_train = torch.zeros((input_dimy, batch_size * self.output_size))

		loss_h = Variable(torch.Tensor([0]), requires_grad=True)

		output_ = []
		I_h = []
		predictions = []

		output_spike_sum = torch.zeros(batch_size, self.output_size).to(self.device)

		for this_t in range(input_dimy):

			#input organization
			input_x = torch.zeros([batch_size, input_dimx])
			input_x = input[:, 0, :, ((this_t)% input_dimy)]

			#################   update states  #########################
			# The first layer neuron only does activation.....
			input_mem, input_spikes = mem_update_adp(input_x.to(self.device),
												input_mem.to(self.device),
												self.thr_i, self.decay_neu)
			input_mem_train[this_t] = input_mem.detach().cpu().view(1, -1)
			input_spike_train[this_t] = input_spikes.detach().cpu().view(1, -1)

			h_input = self.i2h(input_spikes.float())
			hidden_mem, hidden_spikes = mem_update_adp(h_input.to(self.device),
													  hidden_mem.to(self.device),
													  self.thr_h, self.decay_neu)
			hidden_mem_train[this_t] = hidden_mem.detach().cpu().view(1, -1)
			hidden_spike_train[this_t] = hidden_spikes.detach().cpu().view(1, -1)

			output_inputs = self.h2o(hidden_spikes.to(self.device))
			output_mem, output_spikes = mem_update_adp(output_inputs.to(self.device),
													  output_mem.to(self.device),
													  self.thr_o, self.decay_neu)
			output_mem_train[this_t] = output_mem.detach().cpu().view(1, -1)
			output_spike_train[this_t] = output_spikes.detach().cpu().view(1, -1)

			output_spike_sum[:, :] = output_spike_sum[:, :].to(self.device) + output_spikes.to(self.device)

		#################   return spikes sum  #########################
		return output_spike_sum     #, input_spike_train, hidden_spike_train, output_spike_train



class FlyModel_snn(mlpsnn):
	"""
		Inherits from mlpsnn
		add hidden -> hidden recurrent inhibition
		add other weights positive
	"""

	def __init__(self, n_inputs, hidden_dim, n_outputs,
				 decay_neu, noise_lvl, lr, learning_rule,
				 w_decay, h2o_init_zero, device, seed, h2h_inhib_factor=1.0):
		super(FlyModel_snn, self).__init__(n_inputs, hidden_dim, n_outputs, decay_neu, device, seed)

		# learning rate for STDP
		self.lr = lr
		# noise level added to membrane
		self.noise_lvl = noise_lvl
		# amount of inhibition in hidden layer
		self.h2h_inhib_factor = h2h_inhib_factor
		# hidden -> hidden weight
		self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
		# output -> output weight
		self.o2o = nn.Linear(self.output_size, self.output_size, bias=False)

		self.learning_rule = learning_rule
		self.w_decay = w_decay

		if self.learning_rule == 'stdp':
			# define STDP kernels
			self.causal_STDP_kernel = utils.exp_kernel(tau=5.0, a=1.0)
			self.anticausal_STDP_kernel = utils.exp_kernel(tau=5.0, a=-1.0)

		### weight inits:

		# input -> hidden:
		# all hidden neurons should connect to at least 1 input neuron
		x = np.random.binomial(n=self.n_inputs-1, p=0.2, size=self.hidden_size) + 1
		mask = np.zeros([self.hidden_size, self.n_inputs])
		for i in range(self.hidden_size):
			index = np.random.choice(self.n_inputs, x[i], replace=False)
			# binary weights
			mask[i,index] = np.random.uniform(0,0.2,size=len(index))
			# mask[i,index] = np.random.exponential(scale=1.0,size=len(index))
		self.i2h.weight = nn.Parameter(torch.Tensor(mask))

		# # hard-coded best i2h for 3 inputs
		# mask = np.zeros((7,3))
		# mask[0] = 1
		# mask[1,0] = 1
		# mask[2,1] = 1
		# mask[3,2] = 1
		# mask[4] = [1,1,0]
		# mask[5] = [0,1,1]
		# mask[6] = [1,0,1]
		# self.i2h.weight = nn.Parameter(torch.Tensor(mask))

		# print("self.i2h.weight")
		# print(self.i2h.weight)
		# import matplotlib.pyplot as plt
		# plt.figure()
		# with torch.no_grad():
		#     data = self.i2h.weight.numpy()
		#     plt.imshow(data)
		# plt.show()


		# hidden -> hidden:
		# Fully connected random matrix
		self.h2h.weight = nn.Parameter(torch.ones((self.hidden_size, self.hidden_size)))
		with torch.no_grad():
			self.h2h.weight.fill_diagonal_(0)  # Zero on the diagonal (no self-recurrency of neurons)
			# scale down; maybe randomize this value a bit
			self.h2h.weight *= - h2h_inhib_factor / self.hidden_size
		# print("self.h2h.weight")
		# print(self.h2h.weight)

		# hidden -> output
		if h2o_init_zero:
			with torch.no_grad():
				# init weights to zero (will increase only due to learning)
				self.h2o.weight.zero_()
				logging.info("h2o weights initialized as zero.")
		# print("self.h2o.weight")
		# print(self.h2o.weight)

		# # output -> output
		# self.output2output = np.ones((params['n_output_neurons'], params['n_output_neurons']))
		# np.fill_diagonal(self.output2output, 0)  # Zero on the diagonal
		# # scale down; maybe randomize this value a bit
		# self.output2output *= - params['output_inhibition'] / params['n_output_neurons']

		# # output -> output:
		# # Fully connected random matrix
		# self.o2o.weight = nn.Parameter(torch.ones((self.output_size, self.output_size)))
		# with torch.no_grad():
		# 	self.o2o.weight.fill_diagonal_(0)  # Zero on the diagonal (no self-recurrency of neurons)
		# 	# scale down; maybe randomize this value a bit
		# 	self.o2o.weight *= -1.0 # / self.output_size
		logging.info("o2o inhibition disabled.")
		with torch.no_grad():
			# self.h2o.weight.uniform_(0, 1/self.hidden_size)
			# init weights to zero (will increase only due to learning)
			self.o2o.weight.zero_()


	def forward(self, input, labels=None, learning_enabled=False, normalize_pre_rate=False):
		# init
		input_mem = torch.zeros(self.n_inputs)
		hidden_mem =  torch.zeros(self.hidden_size)
		output_mem =  torch.zeros(self.output_size)
		hidden_spike = torch.zeros(self.hidden_size)
		output_spike = torch.zeros(self.output_size)

		# Feed in the whole sequence
		batch_size, seq_num, input_dimx, input_dimy = input.shape

		# Keep track of the spike train of each layer
		input_spike_train = torch.zeros((input_dimy, batch_size, self.n_inputs))
		hidden_spike_train = torch.zeros((input_dimy, batch_size, self.hidden_size))
		output_spike_train = torch.zeros((input_dimy, batch_size, self.output_size))

		# Keep track of the membrane voltages of each layer
		input_mem_train = torch.zeros((input_dimy, batch_size, self.n_inputs))
		hidden_mem_train = torch.zeros((input_dimy, batch_size, self.hidden_size))
		output_mem_train = torch.zeros((input_dimy, batch_size, self.output_size))

		loss_h = Variable(torch.Tensor([0]), requires_grad=True)

		output_ = []
		I_h = []
		predictions = []

		input_spike_sum = torch.zeros(batch_size, self.n_inputs).to(self.device)
		hidden_spike_sum = torch.zeros(batch_size, self.hidden_size).to(self.device)
		output_spike_sum = torch.zeros(batch_size, self.output_size).to(self.device)
		hidden_spikes = None
		output_spikes = None
		
		# logging.warning("Only providing input to first neuron!")

		for this_t in range(input_dimy):

			#input organization
			input_x = torch.zeros([batch_size, input_dimx])
			input_x = input[:, 0, :, ((this_t)% input_dimy)]

			# # only give input to first neuron
			# if this_t < input_dimy // 2:
			#     with torch.no_grad():
			#         # input_x[:,1:] = 0.0
			#         pass
			# else:
			#     with torch.no_grad():
			#         input_x[:,1] = input_x[:,0]
			#         # input_x[:,[0,2]] = 0.0


			#################   update states  #########################
			# The first layer neuron only does activation.....
			input_mem, input_spikes = noisy_mem_update_adp(input_x.to(self.device),
												input_mem.to(self.device),
												self.thr_i, self.decay_neu, self.noise_lvl)
			input_mem_train[this_t] = input_mem.detach().cpu()
			input_spike_train[this_t] = input_spikes.detach().cpu()

			h_input = self.i2h(input_spikes.float())
			# add recurrent inhibition to hidden layer
			if hidden_spikes is not None:
				h_input += self.h2h(hidden_spikes.float())
			hidden_mem, hidden_spikes = noisy_mem_update_adp(h_input.to(self.device),
													  hidden_mem.to(self.device),
													  self.thr_h, self.decay_neu, self.noise_lvl)
			hidden_mem_train[this_t] = hidden_mem.detach().cpu()
			hidden_spike_train[this_t] = hidden_spikes.detach().cpu()

			output_inputs = self.h2o(hidden_spikes.to(self.device))
			# add recurrent inhibition to hidden layer
			if output_spikes is not None:
				output_inputs += self.o2o(output_spikes.float())
			output_mem, output_spikes = noisy_mem_update_adp(output_inputs.to(self.device),
													  output_mem.to(self.device),
													  self.thr_o, self.decay_neu, self.noise_lvl)
			output_mem_train[this_t] = output_mem.detach().cpu()
			output_spike_train[this_t] = output_spikes.detach().cpu()

			input_spike_sum[:, :] = input_spike_sum[:, :].to(self.device) + input_spikes.to(self.device)
			hidden_spike_sum[:, :] = hidden_spike_sum[:, :].to(self.device) + hidden_spikes.to(self.device)
			output_spike_sum[:, :] = output_spike_sum[:, :].to(self.device) + output_spikes.to(self.device)

		# import matplotlib.pyplot as plt

		# fig, ax = plt.subplots()
		# ax.plot(input.reshape(72,5128).T)
		# # plt.plot(input.reshape(5,1000).T)
		# plt.show()
		# plt.close('all')
		
		# # plot mem traces and spikes
		# fig, axes = plt.subplots(6, sharex=True, figsize=(7, 5))
		# for ax, data in zip(axes[::2], [input_mem_train, hidden_mem_train, output_mem_train]):
		#     ax.set_ylim(-0.1,1.1)
		#     ax.plot(data.view(input_dimy, -1))
		# for ax, data in zip(axes[1::2], [input_spike_train, hidden_spike_train, output_spike_train]):
		#     data = data.view(input_dimy, -1).nonzero()
		#     ax.scatter(data[:,0], data[:,1], marker='.', c='black')
		# plt.tight_layout()
		# plt.show()
		# plt.close('all')

		# # plot spikes
		# fig, axes = plt.subplots(3,  figsize=(11, 8))
		# for ax, data in zip(axes, [input_spike_train, hidden_spike_train, output_spike_train]):
		#     data = torch.sum(data.view(input_dimy, -1), axis=0)
		#     ax.bar(range(len(data)), data)
		#     ax.set_ylabel('spike count')
		# plt.suptitle("label: " + str(labels))
		# plt.tight_layout()
		# plt.savefig(f"label {labels}.png")
		# plt.show()
		# plt.close('all')

		# print("labels:", labels)
		# print("input_spike_train.sum()", input_spike_train.sum())
		# print("hidden_spike_train.sum()", hidden_spike_train.sum())
		# print("output_spike_train.sum()", output_spike_train.sum())
		
		if learning_enabled and self.learning_rule == 'stdp':
			with torch.no_grad():
				logging.info("Applying STDP weight update on hidden to output weights")
				dW = utils.calc_STDP_from_spike_trains(pre_spike_train=hidden_spike_train,
										post_spike_train=output_spike_train,
										causal_STDP_kernel=self.causal_STDP_kernel,
										anticausal_STDP_kernel=self.anticausal_STDP_kernel)
				# torch.set_printoptions(precision=3, linewidth=1000)
				# print("lr * dW:", self.lr * dW)
				self.dW = self.lr * dW
				# print("self.h2o.weight", self.h2o.weight)
				self.h2o.weight += self.lr * dW
		elif learning_enabled and self.learning_rule == 'associative':
			with torch.no_grad():
				logging.info("Applying associative learning on hidden to output weights")
				dW = utils.calc_assoc_from_spike_trains(pre_spike_train=hidden_spike_train,
														labels=labels, n_classes=self.output_size,
														normalize_pre_rate=normalize_pre_rate)
				# add weight decay
				dW -= self.w_decay * self.h2o.weight
				# torch.set_printoptions(precision=3, linewidth=300)
				self.dW = self.lr * dW
				# print("self.h2o.weight", self.h2o.weight)
				self.h2o.weight += self.dW

			



		#################   return spikes sum  #########################
		return input_spike_sum, hidden_spike_sum, output_spike_sum, #, input_spike_train, hidden_spike_train, output_spike_train


	def limit_weights(self, min_weight=-1.0, max_weight=1.0):

		for layer in [self.i2h, self.h2h, self.h2o]:
			weight = layer.weight.data

			if  weight.max() > max_weight or weight.min() < min_weight:
				logging.info(f"Limiting weight {layer} between {min_weight, max_weight}.")
				layer.weight.data = torch.clamp(torch.clamp(weight, max=max_weight), min=min_weight)

	def normalize_weights(self, min_weight=-1.0, max_weight=1.0):

		# normalize each weight independently
		# logging.info("Normalizing weight i2h.")
		# self.i2h.weight.data = normalize_tensor_safe(self.i2h.weight.data, min_weight, max_weight)
		logging.info("Normalizing weight h2o.")
		self.h2o.weight.data = utils.normalize_tensor_safe(self.h2o.weight.data, min_weight, max_weight)


	def quantize_weights(self, quantization_type="linear", limit_weights=False,
											 min_weight=-1.0, max_weight=1.0,
											 base_temp=0.02, scale_factor=0.5, n_bits=4):

		logging.info(f"Quantizing weights i2h, h2h and h2o together. Method: {quantization_type}")
		i2h_weight = self.i2h.weight.data
		h2h_weight = self.h2h.weight.data
		h2o_weight = self.h2o.weight.data

		# Stack both weight tensors along a new dimension
		stacked_weights = torch.cat([self.i2h.weight.data.flatten(),
									 self.h2h.weight.data.flatten(),
									 self.h2o.weight.data.flatten()])

		if not limit_weights:
			min_weight = None
			max_weight = None

		# Quantize them together
		if quantization_type == "exponential":
			quantized_weights = quantizationf.FakeQuantOp_exp.apply(stacked_weights, n_bits, max_weight)
		if quantization_type == "exp_probabilistic":
			raise NotImplementedError
			quantized_weights = quantizationf.FakeQuantOp_exp_probabilistic.apply(stacked_weights,
				base_temp, scale_factor, n_bits, min_weight, max_weight)
		if quantization_type == "linear":
			raise NotImplementedError
			quantized_weights = quantizationf.FakeQuantOp_linear.apply(stacked_weights, n_bits, min_weight, max_weight)
		if quantization_type == "linear_sym":
			quantized_weights = quantizationf.FakeQuantOp_linear_sym.apply(stacked_weights, n_bits, min_weight, max_weight)

		# Split them back
		self.i2h.weight.data = quantized_weights[:i2h_weight.numel()].reshape(i2h_weight.shape)
		self.h2h.weight.data = quantized_weights[i2h_weight.numel():-h2o_weight.numel()].reshape(h2h_weight.shape)
		self.h2o.weight.data = quantized_weights[-h2o_weight.numel():].reshape(h2o_weight.shape)

