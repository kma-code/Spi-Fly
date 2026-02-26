"""
	pyTorch implementatation
	of 3-layer SNN FlyModel
	
	Kevin Max, OIST

	based on code by Shimeng Ye, TU/e
	https://github.com/kma-code/SYNCH_perspective
"""

import os
import sys
import logging
logging.basicConfig(format=f'%(levelname)s: %(message)s', level=logging.INFO)
import yaml
from pathlib import Path
from copy import deepcopy

import math
import argparse
from datetime import date

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchsummary import summary

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pylab import *

import neptune
from neptune.utils import stringify_unsupported

from src import quantizationf, neuronmodel, mlpsnn, generatedataset
from src.utils import *

today = date.today()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  #  Specify which GPUs should be visible and available to PyTorch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

try:
	plt.style.use('matplotlib_style.mplstyle')
except:
	plt.rcParams['text.usetex'] = False
	plt.rc('font', size=10,family='serif')

# path to main path of codebase (this script)
MAIN_PATH = Path(__file__).resolve().parents[0]

############################################

# whether to generate new dataset or load an existing one
LOAD_DATASET = False
RECORD_SPIKE_OUTPUTS = True
SAVE_WEIGHTS = True
SAVE_WEIGHT_FIGS = False
criterion = nn.CrossEntropyLoss()  # because it is a classification
DEBUG = False

#################################################


def parse_experiment_arguments():
	"""
		Parse the arguments for the test and train experiments
	"""

	parser = argparse.ArgumentParser(description='Train 3-layer SNN using BPTT, BPTT with EWC, STDP or associative learning.')
	parser.add_argument('--no-neptune', action='store_true',
					default=False,
					help='Do not connect to neptune.ai to save run data, \
					even if neptune id is found.')
	parser.add_argument('--seed', type=int,
					default=42,
					help='Random seed for rng')
	parser.add_argument('--group_tag', type=str, default=None,
					help='Group tag for neptune.ai')
	parser.add_argument('--dataset_id', type=int, default=None,
					help='Number of dataset to use for training')
	parser.add_argument('--params', type=str, required=True,
						help='Path to the parameter .yaml-file.')
	args = parser.parse_args()

	global PARAMS_PATH
	PARAMS_PATH = Path(args.params).resolve().parents[0]

	params = yaml.safe_load(Path(args.params).read_text())
	logging.info(f"Parameters loaded.")
	return args, params

def train_test_val(params,
				   model,
				   train_loader,
				   test_loader,
				   val_loader,
				   odorant_idx,
				   optimizer,
				   min_weight,
				   max_weight,
				   num_epochs,
				   criterion=criterion,
				   n_bits=4,
				   gamma=0.8,
				   min_lr=0.01,
				   max_lr=0.5,
				   base_temp=0.02,
				   scale_factor=0.5):
	w_tracker = []
	w_tracker_q = []
	acc_train = [0.] * num_epochs

	learning_rule = params["learning_rule"]
	normalize_pre_rate = params["normalize_pre_rate"]
	normalize_inputs = params["normalize_inputs"]

	if params['use_scheduler']:
		scheduler = StepLR(optimizer, step_size=2, gamma=gamma)

	if "continual_learning" in params:
		continual_learning = params["continual_learning"]
	else:
		continual_learning = False

	if continual_learning:
		# for continual learning, we are splitting into binary classification tasks
		assert len(odorant_idx) % 2 == 0, "Number of classes needs be divisible by 2"

		cl_subtasks = torch.arange(len(odorant_idx)).int()
		# shuffle randomly
		cl_subtasks = cl_subtasks[torch.randperm(len(odorant_idx))]
		# arrange into pairs
		cl_subtasks = cl_subtasks.reshape(-1,2)
		if continual_learning == 'offline':
			logging.info(f"Offline continual learning enabled.")
		else:
			logging.info(f"Continual learning enabled.")

		tmp_str = f"Number of epochs ({num_epochs}) needs to be multiple of number of classes/2 ({len(odorant_idx)//2})"
		assert num_epochs % (len(odorant_idx)//2) == 0, tmp_str
		epochs_per_subtask = num_epochs // (len(odorant_idx)//2)
		
		logging.info(f"Epochs per subtask: {epochs_per_subtask}")
		cl_subtasks = torch.repeat_interleave(cl_subtasks, epochs_per_subtask, dim=0)
		logging.info(f"Binary subtasks: {cl_subtasks.tolist()}")

		tmp_str = f"Training batch size ({params['batch_size_train']}) needs to be same as number of classes ({len(odorant_idx)})"
		assert params["batch_size_train"] == len(odorant_idx), tmp_str

	if "enable_mixtures" in params and params["enable_mixtures"]:
		logging.info(f"Mixtures enabled: adding random odor with convex scaling {params['mixture_scale']} to every test sample.")
		enable_mixtures = params["enable_mixtures"]
		mixture_scale = params["mixture_scale"]
		assert params["batch_size_valtest"] > 1, "Odors are mixed across batches; batch_size_valtest needs to be > 1"
	else:
		enable_mixtures = False
		mixture_scale = 0.0

	# test on all labels
	# (will be modified later for continual learning)
	subtask_test_labels = None

	if learning_rule == 'EWC_BPTT':
		last_task_EWC = None
		EWC_regularizer_list = []
		old_task_traces = None
		curr_task_traces = torch.Tensor([])
		old_task_labels = None
		curr_task_labels = torch.Tensor([])

	
	train_acc_arr = []
	test_acc = []
	val_acc = []
	for epoch in range(num_epochs):
		train_acc = 0.0
		train_loss_sum = 0.0
		predictions = []
		data_size = 0
		temp_test = 0

		if continual_learning and not continual_learning == 'offline':
			subtask_train_labels = cl_subtasks[epoch].unique()
			subtask_test_labels = cl_subtasks[:epoch+1].unique().ravel()
			logging.info(f"Subtask for epoch {epoch}: labels {subtask_train_labels.tolist()}.")
			if params["learning_rule"] in ['BPTT', 'EWC_BPTT'] and epoch % epochs_per_subtask == 0 and epoch > 0:
				logging.info(f"Resetting optimizer")
				base_params = [model.i2h.weight, model.h2o.weight]
				if params["learning_rule"] == 'BPTT':
					optimizer = torch.optim.Adam([{'params': base_params},],lr=params["lr"],)
				elif params["learning_rule"] == 'EWC_BPTT':
					# optimizer = torch.optim.SGD([{'params': base_params},],lr=params["lr"],)
					optimizer = torch.optim.Adam([{'params': base_params},],lr=params["lr"],)
		elif continual_learning == 'offline':
			subtask_train_labels = cl_subtasks[:epoch+1].unique().ravel()
			subtask_test_labels = cl_subtasks[:epoch+1].unique().ravel()
			logging.info(f"Subtask for epoch {epoch}: labels {subtask_train_labels.tolist()}.")

			if epoch % epochs_per_subtask == 0 and epoch > 0:
				logging.info(f"Re-initializing model")

				if params["learning_rule"] in ['BPTT', 'EWC_BPTT']:
					model = mlpsnn.FlyModel_snn(n_inputs=model.n_inputs,
										  hidden_dim=model.hidden_size,
										  n_outputs=model.output_size,
										  decay_neu=model.decay_neu,
										  noise_lvl=model.noise_lvl,
										  lr=model.lr,
										  learning_rule=model.learning_rule,
										  w_decay=model.w_decay,
										  h2o_init_zero=False,
										  device=device,
										  seed=model.seed,
										  h2h_inhib_factor=model.h2h_inhib_factor)
				elif params["learning_rule"] in ['stdp', 'associative']:
					model = mlpsnn.FlyModel_snn(n_inputs=model.n_inputs,
										  hidden_dim=model.hidden_size,
										  n_outputs=model.output_size,
										  decay_neu=model.decay_neu,
										  noise_lvl=model.noise_lvl,
										  lr=model.lr,
										  learning_rule=model.learning_rule,
										  w_decay=model.w_decay,
										  h2o_init_zero=True,
										  device=device,
										  seed=model.seed,
										  h2h_inhib_factor=model.h2h_inhib_factor)
				model.to(device)

				if params["learning_rule"] == 'BPTT':
					logging.info(f"Resetting optimizer")
					base_params = [model.i2h.weight, model.h2o.weight]
					optimizer = torch.optim.Adam([{'params': base_params},],lr=params["lr"],)
				elif params["learning_rule"] == 'EWC_BPTT':
					logging.info(f"Resetting optimizer")
					base_params = [model.i2h.weight, model.h2o.weight]
					# optimizer = torch.optim.SGD([{'params': base_params},],lr=params["lr"],)
					optimizer = torch.optim.Adam([{'params': base_params},],lr=params["lr"],)
		else:
			subtask_test_labels = None

		if epoch == 0:
			if DEBUG:
				logging.warning("Testing and validation disabled")
			else:
				logging.info("Validating model before training")
				temp_val, _, _, _, _ = test_model(params, model, val_loader, odorant_idx, device, subtask_test_labels, mixture_scale)
				logging.info(f'Validation acc before training: {temp_val}')
				val_acc.append(temp_val)
				logging.info("Testing model before training")
				temp_test, _, _, _, _= test_model(params, model, test_loader, odorant_idx, device, subtask_test_labels, mixture_scale)
				logging.info(f'Test acc before training: {temp_test}')
				test_acc.append(temp_test)
				if run is not None:
					run["val acc"].append(temp_val)
					run["test acc"].append(temp_test)

		labels_arr = []
		input_spike_sum_arr = []
		hidden_spike_sum_arr = []
		output_spike_sum_arr = []

		if learning_rule == 'associative' and normalize_pre_rate:
			logging.info("Normalizing pre-synaptic firing rates for associative learning rule.")

		# for EWC, we calculate the penalty loss
		if learning_rule == 'EWC_BPTT' and epoch % epochs_per_subtask == 0 and epoch > 0 and model.EWC_lambda != 0.0:
			logging.info(f"Initializing EWC regularizer for task {(epoch // epochs_per_subtask)-1}")
			# old_tasks = random.sample(EWC_traces, k=sample_size)
			# instantiate the EWC regularizer using data from the most recent task
			last_task_EWC = EWC_regularizer(model=model, traces=old_task_traces[-1], labels=old_task_labels[-1])
			EWC_regularizer_list.append(last_task_EWC)
		
		for i, (traces, labels) in enumerate(train_loader):
			if DEBUG:
				if i > 1:
					break

			labels = labels[:, 1]
			labels = [odorant_idx.index(element) for element in labels]
			labels = torch.tensor(labels)

			if continual_learning:
				subtask_idx = torch.tensor([torch.where(labels == subtask_label)[0] for subtask_label in subtask_train_labels]).ravel()
				labels = labels[subtask_idx]
				labels = labels.view((-1,len(subtask_train_labels))).long().to(device)
				traces = traces[subtask_idx]
				batch_size = traces.shape[0]
			else:
				batch_size = traces.shape[0]
				labels = labels.view((-1,batch_size)).long().to(device)

			data_size += batch_size

			if normalize_inputs and labels[0,0].item() != 0:
				logging.warning("Normalizing all odors except for class 0")
				tmp_str = f"Normalizing inputs within batch across time and input neurons"
				tmp_str += f" to {norm_factor} volt before amplification."
				logging.info(tmp_str)
				logging.debug(f"traces {traces.mean(), traces.std()}")
				# mean along time axis, norm along neurons
				traces = traces / torch.norm(torch.mean(traces, axis=-1), dim=-1) * norm_factor
				logging.debug(f"normalized traces {traces.mean(), traces.std()}")

			traces = traces * -1 * params['amp']
			traces = traces.unsqueeze(1)
			traces = traces.to(device)

			# if enable_mixtures:
			# 	# randomly shuffle batch
			# 	admix_traces = traces[torch.randperm(traces.size()[0])]
			# 	# make sure that admixing odor is different
			# 	traces = (1.0 - mixture_scale) * traces + mixture_scale * admix_traces

			tmp = [l.item() for l in labels[0]]
			logging.info(f"Training batch with labels: {tmp}")

			# Clear gradients w.r.t. parameters
			if learning_rule in ['BPTT', 'EWC_BPTT']:
				optimizer.zero_grad()

			# w_tracker.append(torch.cat([model.i2h.weight.data.flatten(), model.h2o.weight.data.flatten()]).tolist())

			if params['limit_weights']:
				model.limit_weights(min_weight=min_weight, max_weight=max_weight)
			if params['use_normalization']:
				model.normalize_weights(min_weight=min_weight, max_weight=max_weight)
			if params['use_quantization']:
				model.quantize_weights(quantization_type=params["quantization_type"], limit_weights=params['limit_weights'],
										min_weight=min_weight, max_weight=max_weight, base_temp=base_temp, scale_factor=scale_factor, n_bits=n_bits)

			if SAVE_WEIGHT_FIGS:
				fig = plt.figure()
				with torch.no_grad():
					data = model.h2o.weight.numpy()
					plt.imshow(data, vmin=min_weight, vmax=max_weight)
					plt.colorbar(orientation='horizontal')
					title = f'h2o before epoch {epoch}, batch {i}'
					if params['limit_weights']:
						title +=  ', lim weight'
					if params['use_normalization']:
						title +=  ', norm'
					if params['use_quantization']:
						title +=  ', quant'
					plt.title(title)
				plt.savefig(PARAMS_PATH / f'h2o_epoch{epoch}_batch{i}.png')
				if run is not None:
					run["weight/h2o/img"].append(fig)
				plt.close()


			# w_tracker_q.append(torch.cat([model.i2h.weight.data.flatten(), model.h2o.weight.data.flatten()]).tolist())

			# run training
			input_spike_sum, hidden_spike_sum, output_spike_sum = model.forward(traces, labels=labels, learning_enabled=True,
																				normalize_pre_rate=normalize_pre_rate) # binary input spikes
			input_spike_sum_arr.append(input_spike_sum.detach().cpu().numpy())
			hidden_spike_sum_arr.append(hidden_spike_sum.detach().cpu().numpy())
			output_spike_sum_arr.append(output_spike_sum.detach().cpu().numpy())
			labels_arr.append([l.item() for l in labels[0]])

			#################   classification  #########################
			loss_h = criterion(output_spike_sum.to(device), labels[0].to(device))

			if learning_rule == 'EWC_BPTT':
				if epoch % epochs_per_subtask == 0:
					# it it's the first epoch in a subtask, record dataset
					curr_task_traces = torch.cat((curr_task_traces, traces), dim=0)
					curr_task_labels = torch.cat((curr_task_labels, labels), dim=0)
				if EWC_regularizer_list and model.EWC_lambda != 0.0:
					# for each sub-task, there is one entry in EWC_regularizer_list
					for i, current_EWC in enumerate(EWC_regularizer_list):
						# calculate penalty using current parameters
						EWC_loss = model.EWC_lambda * current_EWC.penalty(new_model=model, task_id=i)
						# add penalty to loss
						loss_h += EWC_loss

			# Getting gradients w.r.t. parameters
			if learning_rule in ['BPTT', 'EWC_BPTT']:
				loss_h.sum().backward()         # retain_graph=True
			loss_h.sum()         # retain_graph=True
			train_loss_sum += loss_h.detach().cpu().numpy()

			if learning_rule in ['BPTT', 'EWC_BPTT']:
				# torch.set_printoptions(precision=12)
				# if current_EWC is not None:
				# 	print("----- pre")
				# 	for n, p in current_EWC._old_params.items():
				# 		print(n, p. p.mean())
				optimizer.step()
				# if current_EWC is not None:
				# 	print("----- post")
				# 	for n, p in current_EWC._old_params.items():
				# 		print(n, p.mean())

			if continual_learning:
				# restrict predictions to available output neurons
				subtask_output_spike_sum = output_spike_sum[:,subtask_idx]
				# need to convert test labels to output neuron id
				subtask_labels = torch.arange(len(subtask_idx))
				pred_ = subtask_output_spike_sum.argmax(axis=1)
				predicted = pred_.t()
				train_acc += (predicted == subtask_labels).sum()
			else:
				pred_ = output_spike_sum.argmax(axis=1)
				predictions.append(pred_.data.cpu().numpy())
				predicted = pred_.t()
				train_acc += (predicted == labels).sum()

			# plt.figure()
			# with torch.no_grad():
			# 	data = output_spike_sum.ravel()
			# 	plt.scatter(range(len(data)), data)
			# 	plt.ylim(-1,)
			# 	plt.ylabel("spike count")
			# 	plt.xlabel("output neuron id")
			# 	plt.title(f'output neuron spikes epoch {epoch}, sample {i}, label {labels_arr[-1]}')
			# plt.savefig(f'tmp/out_spikes_epoch{epoch}_sample{i}_label{labels_arr[-1]}.png')
			# plt.close()		

		# for EWC: save current traces and labels
		if learning_rule == 'EWC_BPTT':
			if old_task_traces is None:
				# append first task
				old_task_traces = curr_task_traces.clone().unsqueeze(0)
				old_task_labels = curr_task_labels.clone().unsqueeze(0)
			else:
				old_task_traces = torch.cat((old_task_traces, curr_task_traces.unsqueeze(0)), dim=0)
				old_task_labels = torch.cat((old_task_labels, curr_task_labels.unsqueeze(0)), dim=0)

			curr_task_traces = torch.Tensor([])
			curr_task_labels = torch.Tensor([])

		w_tracker.append(torch.cat([model.i2h.weight.data.flatten(), model.h2o.weight.data.flatten()]).tolist())

		if params['limit_weights']:
			model.limit_weights(min_weight=min_weight, max_weight=max_weight)
		if params['use_normalization']:
			model.normalize_weights(min_weight=min_weight, max_weight=max_weight)
		if params['use_quantization']:
			model.quantize_weights(quantization_type=params["quantization_type"],
									min_weight=min_weight, max_weight=max_weight, base_temp=base_temp, scale_factor=scale_factor, n_bits=n_bits)

		w_tracker_q.append(torch.cat([model.i2h.weight.data.flatten(), model.h2o.weight.data.flatten()]).tolist())

		# unbatch recorded data
		labels_arr = flatten_nested_list(labels_arr)
		input_spike_sum_arr = flatten_nested_list(input_spike_sum_arr).reshape(-1,model.n_inputs)
		hidden_spike_sum_arr = flatten_nested_list(hidden_spike_sum_arr).reshape(-1,model.hidden_size)
		output_spike_sum_arr = flatten_nested_list(output_spike_sum_arr).reshape(-1,model.output_size)

		if RECORD_SPIKE_OUTPUTS:
			plot_output_spikes_with_lables("train", labels_arr, output_spike_sum_arr, model, epoch, run, PARAMS_PATH)

		# for class_id in np.unique(labels_arr):

		# 	idx = np.where(labels_arr==class_id)[0]
		# 	# print(idx)
		# 	input_spike_sums = input_spike_sum_arr[idx]
		# 	# print("input_spike_sums", input_spike_sums.shape)

		# 	plt.figure()
		# 	with torch.no_grad():
		# 		data = input_spike_sums
		# 		plt.boxplot(data)
		# 		# plt.ylim(0,100)
		# 		plt.ylabel("spike count")
		# 		plt.xlabel("neuron")
		# 		plt.title(f'input neuron spikes for label {class_id}')
		# 		# plt.legend()
		# 	plt.savefig(f'tmp/input_spikes_label{class_id}.png', bbox_inches='tight')
		# 	plt.close()

		# for class_id in np.unique(labels_arr):

		# 	idx = np.where(labels_arr==class_id)[0]
		# 	# print(idx)
		# 	hidden_spike_sums = hidden_spike_sum_arr[idx]
		# 	# print("hidden_spike_sums", hidden_spike_sums.shape)

		# 	plt.figure()
		# 	with torch.no_grad():
		# 		data = hidden_spike_sums
		# 		plt.boxplot(data)
		# 		# plt.ylim(0,100)
		# 		plt.ylabel("spike count")
		# 		plt.xlabel("neuron")
		# 		plt.title(f'hidden neuron spikes for label {class_id}')
		# 		# plt.legend()
		# 	plt.savefig(f'tmp/hidden_spikes_label{class_id}.png', bbox_inches='tight')
		# 	plt.close()

		if DEBUG:
			logging.warning("Testing and validation disabled")
		else:
			temp_val, _, _, val_output_spike_sum_arr, val_labels_arr = test_model(params,model,val_loader,
																					odorant_idx,device, subtask_test_labels, mixture_scale)
			if RECORD_SPIKE_OUTPUTS:
				plot_output_spikes_with_lables("val", val_labels_arr, val_output_spike_sum_arr, model, epoch, run, PARAMS_PATH)
			val_acc.append(temp_val)

			temp_test, _, _, test_output_spike_sum_arr, test_labels_arr = test_model(params,model,test_loader,
																					odorant_idx,device, subtask_test_labels, mixture_scale)
			if RECORD_SPIKE_OUTPUTS:
				plot_output_spikes_with_lables("test", test_labels_arr, test_output_spike_sum_arr, model, epoch, run, PARAMS_PATH)
			test_acc.append(temp_test)

			train_acc_np = train_acc.data.cpu().numpy()/data_size
			acc_train[epoch] = train_acc_np
			if params['use_scheduler']:
				scheduler.step()
			tmp_str = 'epoch: {:2d}'.format(epoch)
			tmp_str +=' Train loss: {:.4f}'.format(train_loss_sum.item()/data_size)
			tmp_str +=' Train acc: {:.4f}'.format(train_acc_np)
			tmp_str +=' Current val_accuracy: {:.4f}'.format(temp_val)
			tmp_str +=' Current test_accuracy: {:.4f}'.format(temp_test)
			logging.info(tmp_str)
			if run is not None:
				run["train loss"].append(train_loss_sum.item()/data_size)
				run["train acc"].append(train_acc_np)
				run["val acc"].append(temp_val)
				run["test acc"].append(temp_test)
			train_acc_arr.append(train_acc_np)

	return train_acc_arr, val_acc, test_acc, w_tracker, w_tracker_q


def run_dataset(params, dataset_id): 
	torch.manual_seed(seed)
	# log params with neptune
	if run is not None:
		run["parameters"] = stringify_unsupported(params)
		run["seed"] = seed
		run["model"] = params["learning_rule"]

	logging.info(f"Parameters: {params}")


	train_loader, test_loader, val_loader, dataset_dict = get_dataloaders(dataset_id, params["ratio1"], params["ratio2"],
																		  params["batch_size_train"],
																		  params["batch_size_valtest"],
																		  params["exclude_SFR"],
																		  seed=seed,
																		  sequential_presentation=params["sequential_presentation"])

	# Extract odorant names and indices
	odorant_name = dataset_dict['odorant_names']
	odorant_idx = dataset_dict['odorant_idx']

	dt = dataset_dict['dt']
	output_dt = dataset_dict['output_dt']
	down_sampling_rate = output_dt / dt

	total_timesteps = dataset_dict['total_steps'] / down_sampling_rate
	for temp_loader in [train_loader, test_loader, val_loader]:
		traces, labels = next(iter(temp_loader))
		assert total_timesteps == traces.size()[-1]

	timestamps = np.arange(0, total_timesteps * output_dt, output_dt)

	dimx = dataset_dict['N_OR'] # n_ORs
	dimy = int(total_timesteps) # number of timesteps
	n_classes = dataset_dict['N_odorants']
	logging.info(f'dimx = {dimx}, dimy = {dimy}, n_classes = {n_classes}')

	test_acc = []

	w_tracker = []
	w_tracker_q = []

	if params["learning_rule"] == 'stdp':
		n_outputs = params["output_dim"]
	elif params["learning_rule"] in ['BPTT', 'associative', 'EWC_BPTT']:
		if params["output_dim"] != n_classes:
			logging.info(f"Setting number of output neurons to number of classes ({n_classes})")
		n_outputs = n_classes
	else:
		raise NotImplementedError("Learning rule must be 'BPTT', 'EWC_BPTT' 'stdp' or 'associative'.")

	if "h2h_inhib_factor" in params:
		h2h_inhib_factor = params["h2h_inhib_factor"]
	else:
		h2h_inhib_factor = 1.0

	if params["learning_rule"] in ['BPTT', 'EWC_BPTT']:
		model = mlpsnn.FlyModel_snn(n_inputs=dimx,
							  hidden_dim=params["hidden_dim"],
							  n_outputs=n_outputs,
							  decay_neu=params["decay_neu"],
							  noise_lvl=params["noise_lvl"],
							  lr=params["lr"],
							  learning_rule=params["learning_rule"],
							  w_decay=params["w_decay"],
							  h2o_init_zero=False,
							  device=device,
							  seed=seed,
							  h2h_inhib_factor=h2h_inhib_factor)
	elif params["learning_rule"] in ['stdp', 'associative']:
		model = mlpsnn.FlyModel_snn(n_inputs=dimx,
							  hidden_dim=params["hidden_dim"],
							  n_outputs=n_outputs,
							  decay_neu=params["decay_neu"],
							  noise_lvl=params["noise_lvl"],
							  lr=params["lr"],
							  learning_rule=params["learning_rule"],
							  w_decay=params["w_decay"],
							  h2o_init_zero=True,
							  device=device,
							  seed=seed,
							  h2h_inhib_factor=h2h_inhib_factor)

	
	if params["learning_rule"] == 'EWC_BPTT':
		# EWC parameter
		model.EWC_lambda = params["EWC_lambda"]

	model.to(device)
	logging.info(f"Device: {device}, {torch.cuda.is_available()}")
	
	base_params = [model.i2h.weight, model.h2o.weight]
	if params["learning_rule"] == 'BPTT':
		optimizer = torch.optim.Adam([{'params': base_params},],lr=params["lr"],)
	elif params["learning_rule"] == 'EWC_BPTT':
		# optimizer = torch.optim.SGD([{'params': base_params},],lr=params["lr"],)
		optimizer = torch.optim.Adam([{'params': base_params},],lr=params["lr"],)
	else:
		optimizer = None

	min_lr = params["min_lr"]
	max_lr = params["max_lr"]
	min_weight = params["min_weight"]
	max_weight = params["max_weight"]
	base_temp = params["base_temp"]
	scale_factor = params["scale_factor"]
	n_bits = params["n_bits"]

	if params['limit_weights']:
		model.limit_weights(min_weight=min_weight, max_weight=max_weight)
	if params['use_normalization']:
		model.normalize_weights(min_weight=min_weight, max_weight=max_weight)
	if params['use_quantization']:
		model.quantize_weights(quantization_type=params["quantization_type"],
								min_weight=min_weight, max_weight=max_weight, base_temp=base_temp, scale_factor=scale_factor, n_bits=n_bits)
	else:
		logging.warning("Weight quantization disabled")

	train_acc, val_acc, test_acc, w_tracker, w_tracker_q = train_test_val(params,
												  model,
												  train_loader,
												  test_loader,
												  val_loader,
												  odorant_idx,
												  optimizer,
												  min_weight, max_weight,
												  num_epochs=params["num_epochs"],
												  criterion=criterion,
												  n_bits=n_bits,
												  gamma=params['gamma'],
												  min_lr=min_lr, max_lr=max_lr,
												  base_temp=base_temp,
												  scale_factor=scale_factor)

	# if run is not None:
	# 	run["test_acc"] = test_acc
	if not DEBUG:
		logging.info(f"Final test accuracy for seed {seed}: {test_acc[-1]}")

	acc_dict = {
	"train acc": train_acc,
	"val acc": val_acc,
	"test acc": test_acc
	}

	w_tracker_dict = {
	"w_tracker": w_tracker,
	"w_tracker_q": w_tracker_q
	}

	return model, acc_dict, w_tracker_dict


def main(rng_seed=42, no_neptune=False, group_tag=None, dataset_id=None, params=None):
	# reset logging with seed numer
	# Remove all handlers associated with the root logger object
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(format=f'Seed {rng_seed} -- %(levelname)s: %(message)s',
					level=logging.INFO)

	if dataset_id is None:
		dataset_id = params["dataset_id"]
		logging.info(f"Loading dataset {dataset_id}")

	# connect to neptune.ai
	global run
	try:
		if not no_neptune:
			with open(MAIN_PATH / 'neptune_id.json') as file:
				id_data = json.load(file)
				project = id_data["project"]
				api_token = id_data["api_token"]
				mode = id_data["mode"] if "mode" in id_data else "async"
				logging.info(f"Neptune.ai id found, connecting to project {project}")
				run = neptune.init_run(
					project=project,
					api_token=api_token,
					mode=mode
					)
			if group_tag is not None:
				run["sys/group_tags"].add(group_tag)
			if dataset_id is not None:
				params["dataset_id"] = dataset_id
				run["dataset_id"] = dataset_id
		else:
			run = None

	except FileNotFoundError:
		logging.info(f"No Neptune.ai id found")
		run = None

	# Generate or load dataset
	if not LOAD_DATASET:
		generatedataset.generate_dataset(dataset_id=dataset_id)

	global seed
	seed = rng_seed

	model, acc_dict, w_tracker_dict = run_dataset(params, dataset_id)

	MODEL_FILE_PATH = PARAMS_PATH / "model.pth"
	logging.info(f"Saving model to {MODEL_FILE_PATH}")
	torch.save(model.state_dict(), MODEL_FILE_PATH)

	ACCS_FILE_PATH = PARAMS_PATH / "accuracies.pkl"
	logging.info(f"Saving accuracies to {ACCS_FILE_PATH}")
	with open(ACCS_FILE_PATH, 'wb') as file:
		dill.dump(acc_dict, file)

	if SAVE_WEIGHTS:
		W_TRACKER_FILE_PATH = PARAMS_PATH / "w_tracker.pkl"
		logging.info(f"Saving w_tracker to {W_TRACKER_FILE_PATH}")
		with open(W_TRACKER_FILE_PATH, 'wb') as file:
			dill.dump(w_tracker_dict, file)

	if run is not None:
		run.stop()

if __name__ == '__main__':
	args, params = parse_experiment_arguments()
	main(rng_seed=args.seed,
		 no_neptune=args.no_neptune,
		 group_tag=args.group_tag,
		 dataset_id=args.dataset_id,
		 params=params)
