import numpy as np
from torch.utils.data.dataset import Dataset

from scipy.signal import butter,filtfilt

import pathlib
CWD = pathlib.Path(__file__).parent

# import labels and receptor response data from DoOR 2.0.1
receptor_labels = np.loadtxt(CWD / "DoOR_datasets/receptor_labels.txt", dtype='str', delimiter=',')
odorant_labels = np.genfromtxt(CWD / "DoOR_datasets/odorant_labels.txt", dtype='str', delimiter='\n')
# # For the DoOR datasets, the odorant labels are also provided by their InChIKey
# odorant_labels = np.loadtxt("DoOR_datasets/odorant_labels_InChIKey.txt", dtype='str', delimiter=',')
receptor_responses = np.loadtxt(CWD / "DoOR_datasets/receptor_responses.csv", delimiter=',')

# exclude all receptors which are not Or
Or_receptor_idx = np.where(np.char.find(receptor_labels, 'Or') == 0)[0]
receptor_labels = receptor_labels[Or_receptor_idx]
receptor_responses = receptor_responses[:,Or_receptor_idx]

def butter_lowpass_filter(data, cutoff, fs, order):
	normal_cutoff = cutoff / (0.5 * fs)
	# Get the filter coefficients 
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = filtfilt(b, a, data)
	return y

class SynthDataset(Dataset):
	"""
		Generates synthetic dataset of ORN voltage traces responding to odorants.
		Every ORN has one type of receptor, which reacts with a given specificity to a given odorant.
		
		Args:
			odorant_idx: choose "all" if all odorants can be picked for simulation, else pass list of indices
			OR_idx: choose "all" to simulate all OR types in the dataset, else pass list of indices
			dt: simulation step width in seconds
			total_steps: how many steps to simulate in total (including noise pre and post stimulus)
			data_steps: how many steps the stimulus should last
			dataset_size: how many ORNs to simulate (total dataset size)
			seed: numpy random seed
			N_ORCOs: how many ORCOs to simulate per ORN
			k_on: opening probability of ORCO ion channel in bin of 5e-5s
			k_off: closing probability of ORCO ion channel in bin of 5e-5s
			noise_mu: mean of baseline noise (in pA)
			noise_sigma: stdev of baseline noise (in pA)
			noise_cutoff: cut-off of low-pass filter for noise (Hz)
			noise_fs: sampling frequency of noise
			conv_open_mean: mean conversion factor from ORCO open state to current in pA
			conv_open_stdev: stdev conversion factor from ORCO open state to current in pA
			conv_closed_mean: mean conversion factor from ORCO closed state to current in pA
			conv_closed_stdev: stdev conversion factor from ORCO closed state to current in pA
			output: "voltage" outputs voltage traces; "mean" outputs mean and std of voltage traces
			R_gap: resistance of ORN/electrode coupling in Ohm
			C_dl: capacitance of electrode in F
			R_t: resistance of electrode in Ohm

		Returns:
			Torch dataset with 'dataset_size' elements of voltage traces of shape: 1 x receptors x total_steps
			or (for output='mean'): 1 x receptors x [mean, stdev]
			and target: [target_id, odorant_id], where target_id is the target class (0, 1, 2 for 3 odorants)
			and odorant_id is the specific id matching the labels in odorant_labels

	"""
	def __init__(self, seed=42, odorant_idx='all', OR_idx='all', dt=5e-5, total_steps=1_000, data_steps=800, dataset_size=100,
				 N_ORCOs=10, k_on = 0.0071, k_off = 0.0966, noise_mu=0.0, noise_sigma=0.5, noise_cutoff=10.0, noise_fs=100.0,
				 conv_open_mean = -1.5, conv_open_stdev = 0.6,
				 conv_closed_mean = 0.0, conv_closed_stdev = 0.3,
				 output='voltage', output_dt=5e-5, R_gap=6.37e+7, C_dl=8.99e-11, R_t=7.92e+8):
		super(SynthDataset, self).__init__()

		# using a numpy RNG to allow compatibility to other deep learning frameworks
		self.rng = np.random.RandomState(seed)

		self.dt = dt
		if self.dt > 1e-2:
			raise ValueError("Simulation step width dt too large, needs to be at most 0.01 s")
		self.output_dt = output_dt
		if self.output_dt < self.dt:
			raise ValueError("Output step width output_dt is smaller than dt, needs to be larger")
		self.total_steps = total_steps
		self.data_steps = data_steps
		# number of total samples to generate
		self.dataset_size = dataset_size

		if isinstance(odorant_idx, list):
			self.odorant_idx = odorant_idx
		elif isinstance(odorant_idx, str) and odorant_idx == 'all':
			self.odorant_idx = np.arange(len(odorant_labels))
		else:
			raise ValueError("odorant_idx should be list or 'all'")

		if isinstance(OR_idx, list):
			self.OR_idx = OR_idx
		elif isinstance(OR_idx, str) and OR_idx == 'all':
			self.OR_idx = np.arange(len(receptor_labels))
		else:
			raise ValueError("OR_idx should be list or 'all'")
			
		self.N_odorants = len(self.odorant_idx)

		# number of ORCOs to simulate per OR type
		self.N_ORCOs = N_ORCOs

		# opening and closing probabilites
		self.k_on = k_on
		self.k_off = k_off

		# noise of ion channel
		self.noise_mu = noise_mu
		self.noise_sigma = noise_sigma
		self.noise_cutoff = noise_cutoff
		self.noise_fs = noise_fs

		# conversion factors from ORCO open/closed to currents; standard values read off of Butterwick et al. 2018 Extended Data Fig. 2d
		self.conv_open_mean = conv_open_mean
		self.conv_open_stdev = conv_open_stdev
		self.conv_closed_mean = conv_closed_mean
		self.conv_closed_stdev = conv_closed_stdev

		# conversion to current:
		self.output = output

		# properties of gap and electrode circuit
		self.R_gap = R_gap
		self.C_dl = C_dl
		self.R_t = R_t
		self.tau = C_dl * R_t
		# convert pA to A
		self.pico_conversion_factor = 1e-12

		self.__vals = []
		self.__cs = []

		for i in range(self.dataset_size):
			print(f"Calculating trace for odor {odorant_idx}")
			# choose class for this sample
			target_id = self.rng.randint(self.N_odorants)
			curr_odorant = self.odorant_idx[target_id]
			sample = self.get_sample(curr_odorant)

			if self.output=='mean':
				sample = np.array([np.mean(sample, axis=-1), np.std(sample, axis=-1)]).T
			self.__vals.append(sample)
			self.__cs.append(np.array([target_id, curr_odorant]))

	def get_sample(self, curr_odorant):

		# the time where the ligand is present (data_steps)
		# will be somewhere within the full measurement time (total_steps).
		# thus, we shift the data randomly within the measurement time

		if self.total_steps == self.data_steps:
			starting_point = 0
		else:
			starting_point = self.rng.randint(0, self.total_steps - self.data_steps)
		# print(f"Od_id {curr_odorant} starting_point {starting_point}")

		# parameters for current class
		k_on = self.k_on
		k_off = self.k_off
		# the channel opening probability scales with the receptor response to the current odorant,
		# which we estimate from the spike rate (after normalization)
		ligand_concentrations = receptor_responses[curr_odorant,self.OR_idx]

		ORN_currents = []
		# for the current odorant, calculate current for every ORN with multiple ORCOs
		# where one ORN is equivalent to its ligand_concentration
		for ligand_concentration in ligand_concentrations:
			# calculate sample including padding
			ORCO_currents = []
			for n in range(self.N_ORCOs):
				ORCO_states = np.zeros(self.data_steps) # start in unbound state
				for i in range(self.data_steps-1):
					ORCO_states[i+1] = self.calc_ORCO_state(ORCO_states[i], k_on, k_off, ligand_concentration, self.dt)

				# calculate current
				current_arr = self.ORCO_to_current(ORCO_states, self.conv_open_mean, self.conv_open_stdev, self.conv_closed_mean, self.conv_closed_stdev)

				# add noise
				noise_arr = self.rng.normal(self.noise_mu, self.noise_sigma, size=self.total_steps)
				noise_arr = butter_lowpass_filter(noise_arr, self.noise_cutoff, self.noise_fs, order=1)

				# move sample to starting point and add to noise
				noise_arr[starting_point:starting_point+len(current_arr)] += current_arr
				current_arr = noise_arr
				
				ORCO_currents.append(np.array(current_arr))
			# sum over ORCOs
			summed_ORCO_currents = np.sum(ORCO_currents, axis=0)
			ORN_currents.append(summed_ORCO_currents)

		ORN_voltages = []
		# model electrode as low-pass filter
		for current in ORN_currents:
			# convert pA current to voltage in Volt
			voltage = 0.0
			voltage_arr = np.zeros(len(current))
			for i, _ in enumerate(current[:-1]):
			    input_current = current[i]
			    voltage += self.dt / self.tau * (- voltage + self.R_gap * input_current * self.pico_conversion_factor)
			    voltage_arr[i+1] = voltage.copy()
			ORN_voltages.append(voltage_arr)

		ORN_voltages = np.array(ORN_voltages)

		# downsample result
		if self.output_dt != self.dt:
			ORN_voltages = ORN_voltages[:,::int(self.output_dt/self.dt)]

		return ORN_voltages


	def __getitem__(self, index):
		sample = [self.__vals[index].copy(), self.__cs[index]]
		return tuple(sample)

	def __len__(self):
		return len(self.__cs)


	def p_open(self, k_on, conc_L, dt):
		"""
			Probability density of ion channel opening
			k_on: binding rate of ligand per dt
			conc_L: ligand concentration
		"""
		return k_on * conc_L

	def ORCO_to_current(self, ORCO_states, open_mean, open_stdev, closed_mean, closed_stdev):
		"""
			Converts the open/closed states to currents
			mean: mean for closed/open current e.g. from Butterwick et al.
			stdev: std dev for closed/open current
		"""
		curr = ORCO_states.astype('float')
		curr[curr == 1] = np.random.normal(open_mean, open_stdev, size=curr[curr == 1].shape)
		curr[curr == 0] = np.random.normal(closed_mean, closed_stdev, size=curr[curr == 0].shape)
		return curr

	def p_close(self, k_off, dt):
		"""
			Probability density of ion channel closing
			k_off: unbinding rate of ligand per dt
		"""
		return k_off

	def calc_ORCO_state(self, curr_state, k_on, k_off, conc_L, dt):
		"""
			Returns updated state of single ORCO
			0: ORCO closed
			1: ORCO open
		"""
		rand = self.rng.random()
		if curr_state == 0:
			if rand < self.p_open(k_on, conc_L, dt):
				return 1
			else:
				return 0
		elif curr_state == 1:
			if rand < self.p_close(k_off, dt):
				return 0
			else:
				return 1
