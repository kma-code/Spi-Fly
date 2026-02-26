import math
import torch
import torch.nn.functional as F
import torchaudio.functional as F2
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset, ConcatDataset
import numpy as np
import os
import json
import dill
from collections import Counter
import itertools
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pylab import *
from pathlib import Path
from copy import deepcopy

# path to main path of codebase
MAIN_PATH = Path(__file__).resolve().parents[1]


class CustomTensorDataset(Dataset):
	def __init__(self, traces, labels):
		assert traces.size(0) == labels.size(0)
		self.tensors = [traces, labels]
		self.traces = traces
		self.labels = labels

	def __getitem__(self, index):
		trace = self.tensors[0][index]
		label = self.tensors[1][index]
		return trace, label

	def __len__(self):
		return self.tensors[0].size(0)


def adjusted_sigmoid(x, x_target=0.6):
	# Solve for s to ensure sigmoid(x_target) = 0.95
	center = x_target / 2
	s = -torch.log(torch.tensor(1/0.95 - 1)) / (x_target - center)
	return 1 / (1 + torch.exp(-s * (torch.abs(x) - center)))

def cal_lr(x, max_val, min_lr=0.01, max_lr=0.5):
	return min_lr + (max_lr - min_lr) * adjusted_sigmoid(x, x_target=max_val)

def pre_update(model, max_val, min, max):
	for i, param in enumerate(model.parameters()):
		if i >= 2:
			lr_matrice = cal_lr(param, max_val, min_lr=min, max_lr=max)
			param.grad.mul_(lr_matrice) 
			#logging.info(f'param is : {param}')
			#logging.info(f'Learning rate is : {lr_matrice}')
			#logging.info(f'new_matrice gradient is : {param.grad}')

def normalize_tensor_safe(tensor, min_weight, max_weight, epsilon=1e-6):
	X_min, X_max = tensor.min(), tensor.max()
	if X_min == X_max:
		logging.info(f"All entries of weight are equal and will not be normalized.")
		return tensor
	elif X_min >= 0.0:
		logging.info(f"All entries of weight are positive. Normalizing between [0,{max_weight}].")
		min_weight = 0.0
	elif X_max <= 0.0:
		logging.info(f"All entries of weight are negative. Normalizing between [{min_weight},0].")
		max_weight = 0.0
	return min_weight + (tensor - X_min) * (max_weight - min_weight) / (X_max - X_min + epsilon)


def test_model(config, model, test_loader, odorant_idx, device,
							 test_labels=None, mixture_scale=0.0):
	test_acc = []
	if test_labels is not None:
		logging.info(f"Validating/testing on labels: {test_labels.tolist()}.")

	labels_arr = []
	input_spike_sum_arr = []
	hidden_spike_sum_arr = []
	output_spike_sum_arr = []
	for i, (traces, labels) in enumerate(test_loader):
		batch_size = traces.shape[0]

		labels = labels[:, 1]
		labels = [odorant_idx.index(element) for element in labels]
		labels = torch.tensor(labels)

		if test_labels is not None:
			# test only on test_labels
			n_test_labels = len(test_labels)
			test_idx = torch.tensor([torch.where(labels == test_label)[0] 
																	for test_label in test_labels]).ravel()
			labels = labels[test_idx]
			labels = labels.view((-1,n_test_labels)).long().to(device)
			traces = traces[test_idx]
		else:
			labels = labels.view((-1,batch_size)).long().to(device)

		traces = traces * -1 * config['amp']
		traces = traces.unsqueeze(1)
		traces = traces.to(device)

		if mixture_scale != 0.0:
			# randomly shuffle batch
			admix_traces = traces[torch.randperm(traces.size()[0])]
			# admix shuffled batch in convex combination
			traces = (1.0 - mixture_scale) * traces + mixture_scale * admix_traces

		# logging.info(f"Current labels: {labels}")
		input_spike_sum, hidden_spike_sum, output_spike_sum = model.forward(traces, labels=labels)
		input_spike_sum_arr.append(input_spike_sum.detach().cpu().numpy())
		hidden_spike_sum_arr.append(hidden_spike_sum.detach().cpu().numpy())
		output_spike_sum_arr.append(output_spike_sum.detach().cpu().numpy())
		labels_arr.append([l.item() for l in labels[0]])

		#################   classification  #########################
		if test_labels is not None:
			# restrict predictions to available output neurons
			output_spike_sum = output_spike_sum[:,test_labels]
			# need to convert test labels to output neuron id
			labels = torch.arange(len(test_labels))
		pred_ = output_spike_sum.argmax(axis=1)

		test_accuracy = (pred_.to(device) == labels.to(device)).sum().data.cpu().numpy() / float(len(pred_))
		# logging.info(f"Batch labels: {labels}, prediction: {pred_}")
		# logging.info('Batch test acc: {:.4f}'.format(test_accuracy))
		test_acc.append(test_accuracy)
	# logging.info('Test accuracy: {:.4f}'.format(np.mean(test_acc)))
	return np.mean(test_acc), input_spike_sum_arr, hidden_spike_sum_arr, output_spike_sum_arr, labels_arr



def get_batch_size_power_of_2(train_size):

	# Calculate the batch size (1/10 of training samples)
	batch_size = int(train_size / 10)

	# Find the largest power of 2 less than or equal to the batch size
	batch_size_lower_power_of_2 = 2 ** (math.floor(math.log2(batch_size)))

	# Check if the 1/10 batch size is more than 70% larger than the lower power of 2
	if batch_size >= 1.7 * batch_size_lower_power_of_2:
		# Find the smallest power of 2 greater than or equal to the batch size
		batch_size_power_of_2 = 2 ** (math.ceil(math.log2(batch_size)))
	else:
		# Use the lower power of 2
		batch_size_power_of_2 = batch_size_lower_power_of_2
	return batch_size_power_of_2


def get_dataloaders(dataset_id, ratio1, ratio2,
					batch_size_train=None, batch_size_valtest=None,
					exclude_SFR=False, seed=42,
					sequential_presentation=False):

	""" 
		Two datasets are available:
		'synth': synthetic dataset based on Max et al., 2025, doi 10.1088/2634-4386/aded2d
		'gas_sensor": gas sensor dataset by Vergara et al., 2013, doi 10.24432/C5JP5N
	"""

	torch.manual_seed(seed)

	if isinstance(dataset_id, int):
		dataset_type = "synth"
	elif dataset_id == "gas_sensor":
		dataset_type = "gas_sensor"
	else:
		raise NotImplementedError("Dataset_type must be 'synth' or 'gas_sensor'")

	if dataset_type == "synth":
		saved_dataset_dir = MAIN_PATH / f"saved_datasets/dataset{dataset_id}/"
		
		# Build the dataset name for the JSON template file
		dataset_name = f"dataset_template_{dataset_id}.json"
		dataset_path = os.path.join(saved_dataset_dir, dataset_name)
		
		# Load the corresponding dict from the dataset template
		with open(dataset_path) as f:
			dataset_dict = json.load(f)

		# Build the file name for the generated voltage data
		voltage_file_name = f"dataset_size{dataset_dict['dataset_size']}_Nodor{dataset_dict['N_odorants']}_NOR{dataset_dict['N_OR']}_voltage.pkl"
		voltage_file_path = os.path.join(saved_dataset_dir, voltage_file_name)

		# Load the generated dataset
		with open(voltage_file_path, 'rb') as f:
			dataset = dill.load(f)

		logging.info(f"Loading dataset_template_{dataset_id}.json -> {voltage_file_path} completed")

	elif dataset_type == "gas_sensor":

		# create dataset dictionary
		dataset_dict =	{
				"odorant_names": [
				"Toluene",
				"Benzene",
				"Methane",
				"CO",
				"Ammonia",
				"Acetone",
				"Acetaldehyde",
				"Methanol",
				"Butanol",
				"Ethylene"
				],
				"odorant_idx": [
				0,
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				9
				],
				"N_OR": 72,
				"dt": 0.005,
				"output_dt": 0.005,
				"dataset_size": 20,
				"N_odorants": 10,
			}

		saved_dataset_dir = MAIN_PATH / f"saved_datasets/gas_sensor_dataset/"

		voltage_file_name = f"dataset_size_20_baseline_subtraction_True_data.pkl"
		voltage_file_path = os.path.join(saved_dataset_dir, voltage_file_name)

		labels_file_name = f"dataset_size_20_baseline_subtraction_True_labels.pkl"
		labels_file_path = os.path.join(saved_dataset_dir, labels_file_name)

		# Load the dataset
		with open(voltage_file_path, 'rb') as f:
			voltages = dill.load(f)
		with open(labels_file_path, 'rb') as f:
			labels = dill.load(f)

		odors_labels = dataset_dict["odorant_names"]
		labels_idx = np.array([np.where(label == np.array(odors_labels))[0] for label in labels])

		voltages = torch.tensor(voltages).permute(0,2,1)
		labels_idx = torch.tensor(labels_idx)

		# the synthetic dataset has an extra label entry, but only the last one is used.
		# example: labels[:,1]
		# for compatibility, we pad the labels of this dataset to the same size:
		tmp_pad = torch.zeros(labels_idx.shape, dtype=int)
		labels_idx = torch.concatenate((tmp_pad, labels_idx), axis=1)

		dataset = CustomTensorDataset(voltages, labels_idx)

		logging.info(f"Loading dataset_params.json -> {voltage_file_path} completed")

		dataset_dict["total_steps"] = voltages.size(-1)


	if exclude_SFR:
		logging.info("Excluding odorant_idx 0 = spontaneous firing rate (SFR) traces from classification")
		logging.warning("Make sure that odor 0 is SFR")
		dataset_dict['N_odorants'] -= 1
		dataset_dict['odorant_idx'].remove(0)
		dataset_dict['odorant_names'].remove("SFR")

	dataset_size_total = dataset_dict['dataset_size'] * dataset_dict['N_odorants']

	# Splitting the dataset into train, validation, and test sizes
	train_size = int(dataset_size_total * ratio1)
	val_size = int(ratio2 * dataset_size_total)
	test_size = dataset_size_total - train_size - val_size  # Ensure all samples are used
	logging.info(f"Total number of train samples is: {train_size}")
	logging.info(f"Total number of validation samples is: {val_size}")
	logging.info(f"Total number of test samples is: {test_size}")

	# split into equal number of samples for each class:
	dataset_train = []
	dataset_val = []
	dataset_test = []

	if sequential_presentation:
		tmp_str  = "Organizing dataset into sequences such that"
		tmp_str += " every sample appears once before repeating"
		logging.info(tmp_str)
		# we will shuffle manually, so no need to shuffle in the dataloaders
		shuffle = False
		rng = np.random.default_rng(seed=seed)

		if exclude_SFR:
			for ds in dataset.datasets:
				if ds.odorant_idx == [0]:
					logging.info("Removed odorant_idx 0 from dataset")
					dataset.datasets.remove(ds)
		N_odorants = dataset_dict['N_odorants']
		# number of sequences (1 sequence = all odorants, once)
		N_seq_total = dataset_dict['dataset_size']
		N_seq_train = int(N_seq_total * ratio1)
		N_seq_val = int(N_seq_total * ratio2)
		N_seq_test = int(N_seq_total * ratio2)

		# we need two random shufflings:
		# - one to determine the order of odors
		# - one to determine which trace to select for a given odor

		# order of odors
		odor_idx = np.arange(N_odorants)[:,np.newaxis]
		odor_idx = np.tile(odor_idx, N_seq_total).T
		# shuffle within axis of odors, not samples
		odor_idx = rng.permuted(odor_idx, axis=1)
		train_odor_idx = odor_idx[:N_seq_train]
		val_odor_idx = odor_idx[N_seq_train:N_seq_train+N_seq_val]
		test_odor_idx = odor_idx[-N_seq_test:]

		# trace selection:
		total_trace_idx = np.arange(N_seq_total)
		total_trace_idx = rng.permuted(total_trace_idx)

		# train
		train_trace_idx = total_trace_idx[:N_seq_train,np.newaxis]
		train_trace_idx = np.tile(train_trace_idx, N_odorants)
		# shuffle within axis of odors, not samples
		train_trace_idx = rng.permuted(train_trace_idx, axis=0)

		# val
		val_trace_idx = total_trace_idx[N_seq_train:N_seq_train+N_seq_val,np.newaxis]
		val_trace_idx = np.tile(val_trace_idx, N_odorants)
		# shuffle within axis of odors, not samples
		val_trace_idx = rng.permuted(val_trace_idx, axis=0)

		# test
		test_trace_idx = total_trace_idx[N_seq_train+N_seq_val:N_seq_total,np.newaxis]
		test_trace_idx = np.tile(test_trace_idx, N_odorants)
		# shuffle within axis of odors, not samples
		test_trace_idx = rng.permuted(test_trace_idx, axis=0)

		dataset_train = get_sequential_dataset(dataset,
												 N_seq_train,
												 train_odor_idx,
												 train_trace_idx,
												 N_odorants,
												 N_seq_total,
												 dataset_type)
		dataset_val   = get_sequential_dataset(dataset,
												 N_seq_val,
												 val_odor_idx,
												 val_trace_idx,
												 N_odorants,
												 N_seq_total,
												 dataset_type)
		dataset_test  = get_sequential_dataset(dataset,
												 N_seq_test,
												 test_odor_idx,
												 test_trace_idx,
												 N_odorants,
												 N_seq_total,
												 dataset_type)



	else:
		logging.info("Enabling shuffling of data")
		shuffle = True
		for ds in dataset.datasets:
			train_indices, val_test_indices, train_label, val_test_label = train_test_split(
																range(len(ds)),
																ds._SynthDataset__cs,
																stratify=ds._SynthDataset__cs,
																test_size=1-ratio1,
																random_state=1
															)
			if exclude_SFR:
				train_indices = np.array(train_indices)[np.where(np.array(train_label)[:,-1] != 0)[0]]
			ds_train = Subset(ds, train_indices)
			ds_val_test = Subset(ds, val_test_indices)

			test_indices, val_indices, test_label, val_label = train_test_split(
																range(len(ds_val_test)),
																val_test_label,
																stratify=val_test_label,
																test_size=0.5,
																random_state=1
															)
			if exclude_SFR:
				val_indices = np.array(val_indices)[np.where(np.array(val_label)[:,-1] != 0)[0]]
				test_indices = np.array(test_indices)[np.where(np.array(test_label)[:,-1] != 0)[0]]
			ds_test = Subset(ds_val_test, test_indices)
			ds_val = Subset(ds_val_test, val_indices)

			dataset_train.append(ds_train)
			dataset_val.append(ds_val)
			dataset_test.append(ds_test)

	dataset_train = ConcatDataset(dataset_train)
	dataset_val = ConcatDataset(dataset_val)
	dataset_test = ConcatDataset(dataset_test)

	if batch_size_train is None:
		batch_size_train = get_batch_size_power_of_2(train_size)
	if batch_size_valtest is None:
		batch_size_valtest = get_batch_size_power_of_2(train_size)

	logging.info(f"Using train batch size: {batch_size_train}")
	logging.info(f"Using val and test batch size: {batch_size_valtest}")

	train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=shuffle)
	test_loader = DataLoader(dataset_test, batch_size=batch_size_valtest, shuffle=shuffle)
	val_loader = DataLoader(dataset_val, batch_size=batch_size_valtest, shuffle=shuffle)

	logging.info("Number of samples per class:")
	for loader, name in zip([train_loader,test_loader,val_loader],
							["train set", "test set", "val set"]):
		counted_labels = Counter()
		for _, (traces, labels) in enumerate(loader):
			counted_labels += Counter(labels[:,1].numpy())
		logging.info(name + ": " + str(dict(counted_labels)))

	return train_loader, test_loader, val_loader, dataset_dict


def get_sequential_dataset(dataset, set_size, set_odor_idx, set_trace_idx,
							 N_odorants, dataset_size, dataset_type):

	seq_dataset = []

	# for every set of odorants
	for i in range(set_size):
		for j in range(N_odorants):
			if dataset_type == 'synth':
				tmp_ds = Subset(dataset.datasets[set_odor_idx[i,j]], [set_trace_idx[i,j]])
			elif dataset_type == 'gas_sensor':
				tmp_ds = Subset(dataset, [set_odor_idx[i,j] * dataset_size + set_trace_idx[i,j]])
			seq_dataset.append(tmp_ds)
	return seq_dataset


def remove_outliers_and_calculate_mean(data):
	# Assuming data is of shape (num_datasets, num_iteration)
	
	# Initialize a list to store means of each dataset after removing outliers
	dataset_means = []
	
	# Loop over each dataset
	for dataset in data:
		# Flatten the dataset across iterations
		dataset = np.concatenate([np.array(item).flatten() for item in dataset])
		
		# Calculate percentiles for the dataset
		Q1 = np.percentile(dataset, 25)  # First quartile (25th percentile)
		Q3 = np.percentile(dataset, 75)  # Third quartile (75th percentile)
		IQR = Q3 - Q1  # Interquartile range

		# Define the bounds for filtering
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR

		# Filter out the outliers
		filtered_data = dataset[(dataset >= lower_bound) & (dataset <= upper_bound)]

		# Calculate the mean of the filtered data for this dataset
		dataset_means.append(np.mean(filtered_data))
	
	# After all datasets, calculate the overall mean of all dataset means (no outlier removal here)
	all_data = np.concatenate([np.array(dataset).flatten() for dataset in data])
	
	overall_mean = np.mean(all_data)
	
	return overall_mean

def calc_STDP_from_spike_trains(pre_spike_train, post_spike_train, causal_STDP_kernel=None, anticausal_STDP_kernel=None):
	"""
		calculates STDP weight update
		given pre- and post-synaptic spike trains

		expects:
			spike trains of size [time_steps,batch_size,n_neurons]
		returns:
			array dW of shape (n_post,n_pre)
	"""

	pre_spike_train = pre_spike_train.permute([1,2,0])
	post_spike_train = post_spike_train.permute([1,2,0])

	batch_size = pre_spike_train.shape[0]
	n_pre = pre_spike_train.shape[1]
	n_post = post_spike_train.shape[1]
	len_spike_trains = pre_spike_train.shape[-1]

	pre_kernel = causal_STDP_kernel.repeat((batch_size, n_pre, 1))
	pre_tracker = F2.convolve(pre_spike_train, pre_kernel)[:,:,:len_spike_trains]
	assert pre_spike_train.shape == pre_tracker.shape

	post_kernel = anticausal_STDP_kernel.repeat((batch_size, n_post, 1))
	post_tracker = F2.convolve(post_spike_train, post_kernel)[:,:,:len_spike_trains]
	assert post_spike_train.shape == post_tracker.shape

	# multiply tracker with spike trains and sum along batches
	dW = torch.matmul(post_spike_train, pre_tracker.permute([0,2,1])).sum(axis=0)
	dW += torch.matmul(post_tracker, pre_spike_train.permute([0,2,1])).sum(axis=0)

	return dW

def exp_kernel(tau=5.0, a=1.0, bias=0.0):
	"""
		Defines a general exponential kernel
		with decay constant tau
		and kernel width of 5 tau
	"""
	kernel = a * torch.exp(-torch.arange(5*tau)/tau) + bias
	# no change for spikes arriving at exactly the same time
	kernel[0] = 0.0
	return kernel

def box_kernel(tau=5.0, a=1.0, bias=0.0):
	"""
		Defines a general box kernel of width tau
	"""
	kernel = a * torch.ones(int(tau)) + bias
	kernel[0] = 0.0
	return kernel

def calc_assoc_from_spike_trains(pre_spike_train, labels, n_classes, normalize_pre_rate=False):
	"""
		calculates associative learning weight update
		given pre-synaptic spike trains and labels

		expects:
			spike train of size [time_steps,batch_size,n_pre]
			labels of size [1,batch_size]
			normalize_pre_rate: whether to normalize the pre-synaptic
								firing rate to 1 (helps equalize odor responses)
		returns:
			array dW of shape (n_post,n_pre)
	"""

	batch_size = pre_spike_train.shape[1]
	# label to one-hot
	post_activation = F.one_hot(labels.long(), n_classes)
	post_activation = post_activation.reshape((batch_size,n_classes)).float()

	# sum pre-synaptic spike train over time
	pre_rate = pre_spike_train.sum(axis=0)

	# normalize across pre-synaptic neurons
	if normalize_pre_rate:
		norm = pre_rate.sum(axis=-1)[:, None]
		# replace 0 with 1 (if no hidden spikes)
		norm[norm == 0.0] = 1.0
		pre_rate = pre_rate / norm

	# outer product / sum across batches
	dW = post_activation.mT @ pre_rate

	return dW

def flatten_nested_list(arr):
	return np.array([[x for xs in arr for x in xs]])

def plot_output_spikes_with_lables(title,labels_arr,
									 output_spike_sum_arr,
									 model,epoch,run,
									 PARAMS_PATH):
	
	# unbatch recorded data
	labels_arr = flatten_nested_list(labels_arr)
	output_spike_sum_arr = flatten_nested_list(output_spike_sum_arr).reshape(-1,model.output_size)
	assert len(labels_arr.ravel()) == len(output_spike_sum_arr)

	# arrange into tuples (label, output_spike_sum)
	labels_spikes_arr = [(lab, spks) for lab, spks in zip(labels_arr.ravel(), output_spike_sum_arr)]
	FILE_PATH = PARAMS_PATH / f'epoch{epoch}_{title}_labels_and_output_spikes.pkl'
	logging.info(f"Saving output spikes with labels to {FILE_PATH}")
	with open(FILE_PATH, 'wb') as file:
		dill.dump(labels_spikes_arr, file)

	# for class_id in np.unique(labels_arr):
	# 	idx = np.where(labels_arr==class_id)[0]
	# 	output_spike_sums = output_spike_sum_arr[idx]

	# 	fig = plt.figure(layout='constrained')
	# 	ax = fig.gca()
	# 	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	# 	with torch.no_grad():
	# 		data = output_spike_sums
	# 		for i, d in enumerate(data.T):
	# 			if run is not None:
	# 				for entry in d:
	# 					run[f"{title}/epoch{epoch}_label{class_id}_neuron{i}/output_spike_sums"].append(entry)
	# 			ax.plot(d, label=f'neuron {i}')
	# 		# plt.ylim(0,100)
	# 		ax.set_ylabel("spike count")
	# 		ax.set_xlabel("presentations")
	# 		fig.suptitle(f'output neuron spikes for label {class_id} ({title})')
	# 		if i <= 5:
	# 			ax.legend()
	# 	plt.savefig(PARAMS_PATH + f'/epoch{epoch}_output_spikes_{title}_label{class_id}.png', bbox_inches='tight')
	# 	if run is not None:
	# 		run[f"output_spike_sums/{title}"].append(fig)
	# 	plt.close()


class EWC_regularizer(object):
	"""
			Elastic weight consolidiation:
			Pytorch implementation adapted from Ryuichiro Hataya
			https://github.com/moskomule/ewc.pytorch
	"""

	def __init__(self, model, traces, labels):

		self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
		self._old_params = {}
		self._fim = self.calculate_fim(model, traces, labels)

		for n, p in deepcopy(self.params).items():
				self._old_params[n] = p.data
				self._old_params[n].requires_grad = False

	def calculate_fim(self, model, traces, labels, empirical=False):
		"""
			Estimates the Fisher matrix from the dataset
			we use the batched Fisher matrix over the full dataset
			either "exact batched" or "empirical batched"
			see https://arxiv.org/pdf/2502.11756
		"""
		labels = labels.flatten()

		fim = {}
		for n, p in deepcopy(self.params).items():
				p.data.zero_()
				fim[n] = p.data

		model.eval()

		# we use the full dataset (instead of sampling)
		# run dataset through network
		# for trace, label in zip(traces, labels):
		model.zero_grad()
		_, _, output_spike_sum = model.forward(traces)
		# output_spike_sum = output_spike_sum.view(1, -1)

		if empirical:
			# logging.info("Calculating empirical Fisher matrix")
			out_dx = labels.type(torch.int64).view(-1)
		else:
			# logging.info("Calculating exact Fisher matrix")
			out_dx = output_spike_sum.argmax(axis=1).view(-1)

		fim_loss = F.nll_loss(F.log_softmax(output_spike_sum, dim=1), out_dx)
		# fim_loss = - torch.nn.functional.log_softmax(output_spike_sum, dim=1).gather(-1, out_dx.view(-1, 1)).mean(0)
		# calculate gradients on old tasks dataset
		# print("--- fim_loss", fim_loss)
		fim_loss.backward()

		for n, p in model.named_parameters():
				fim[n].data = p.grad.data.clone().detach() ** 2 / len(traces)
		model.zero_grad()

		fim = {n: p for n, p in fim.items()}
		return fim

	def penalty(self, new_model, task_id=None):
		if task_id is not None:
			logging.info(f"Calculating EWC penalty for task {task_id}")
		else:
			logging.info("Calculating EWC penalty")
		penalty = 0
		for n, p in new_model.named_parameters():
			_penalty = self._fim[n] * (p - self._old_params[n]) ** 2
			penalty += _penalty.sum()

		return penalty
