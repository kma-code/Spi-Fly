# Original Authors:
# Imam, Nabil  Cleland, Thomas [tac29 at cornell.edu] 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, October 2022, n.dennler2@herts.ac.uk
# 
# Modified by Kevin Max, February 2026, kevin.max@oist.jp

import os
import csv
import pickle
import numpy as np
import random
import copy
from pathlib import Path
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def loadFile(fileName, time_sample=None, sample_dt=1e-3, output_dt=1e-3):
    """
    Load data points of a given data file and point in time

    :param str fileName: filename
    :param int time_sample_onset: onset of sampling time in seconds
    :return list, int: all sensor readings, exact sampling time in seconds
    """
    lines = []
    data = csv.reader(open(fileName, 'r'), delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)

    for i in data:
        lines.append(i[:-1])
    odor_raw = np.array(lines)

    # odor_raw has dimension time [ms] x features.
    # select columns which contain sensor readout:
    # see https://archive.ics.uci.edu/dataset/251/gas+sensor+arrays+in+open+sampling+settings
    # for how these indices are selected
    odor_raw = odor_raw[:,12:]
    no_feature_cols = np.arange(1,9)*9-1
    odor_raw = np.delete(odor_raw, no_feature_cols, axis=1)
    odor_raw = odor_raw.astype(int)

    # if a sample time is given, select only that time
    # else, all time points are returned
    if time_sample is not None:
        time_sample = int(time_sample*1000)
        odor_raw = odor_raw[time_sample:time_sample+1]

    # finally, we downsample the dataset
    if output_dt != sample_dt:
        downsample_factor = int(output_dt/sample_dt)
        if downsample_factor > 1:
            odor_raw = odor_raw[::downsample_factor]

    return odor_raw, time_sample

def find_gas_file(data_dir, gas='CO_4000'):
    """
    In data directory, find gas file that matches a given gas string

    :param str data_dir: data directory
    :param str gas: gas identifying string, defaults to 'CO_1000'
    :return list: data files where string is matched
    """
    data_files = []
    for d in os.listdir(data_dir):
        if d.find(gas) != -1:
            data_files.append(data_dir / d)
    return data_files

def get_gas_files(data_dir, gas, n_samples=10, output_dt=1e-3, time_baseline=1):
    """
    Get n gas files for a given gas and time, plus baseline

    :param str data_dir: data directory
    :param str gas: gas identifying string
    :param int n_samples: number of gas samples per gas, defaults to 10
    :return list, list: gas data, baseline data
    """
    raw_all = []
    baseline_all = []
    
    filenames = find_gas_file(data_dir, gas=gas)
    if len(filenames) < n_samples:
        tmp_str  = f"n_samples {n_samples} is larger than number of files found ({len(filenames)}), "
        tmp_str += f"setting n_samples to {len(filenames)}."
        print(tmp_str)
        n_samples = len(filenames)

    # randomly select n_samples
    for i in range(n_samples):
        filename = random.choice(filenames)
        print(f"Loading {filename}")
        raw, _ = loadFile(filename, output_dt=output_dt)
        baseline, _ = loadFile(filename, time_sample=time_baseline)

        raw_all.append(raw)
        baseline_all.append(baseline)
    
    return raw_all, baseline_all

def loadData(data_dir, n_samples, output_dt, time_baseline):
    """
    Load data. Returns raw data, baseline data and labels

    :param str dir: data directory name
    :param int n_samples: number of gas samples per gas
    :return list, list, list: odors_raw, odors_baseline, odors_labels
    """

    # data_dir = "data/"+dir

    Toluene_raw, Toluene_baseline   = get_gas_files(data_dir, 'Toluene_200', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Benzene_raw, Benzene_baseline   = get_gas_files(data_dir, 'Benzene_200', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Methane_raw, Methane_baseline   = get_gas_files(data_dir, 'Methane_1000', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    CO_raw, CO_baseline             = get_gas_files(data_dir, 'CO_4000', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Ammonia_raw, Ammonia_baseline   = get_gas_files(data_dir, 'Ammonia_10000', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Acetone_raw, Acetone_baseline   = get_gas_files(data_dir, 'Acetone_2500', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Acetaldehyde_raw, Acetaldehyde_baseline = get_gas_files(data_dir, 'Acetaldehyde_500', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Methanol_raw, Methanol_baseline = get_gas_files(data_dir, 'Methanol_200', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Butanol_raw, Butanol_baseline   = get_gas_files(data_dir, 'Butanol_100', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)
    Ethylene_raw, Ethylene_baseline = get_gas_files(data_dir, 'Ethylene_500', n_samples=n_samples, output_dt=output_dt, time_baseline=time_baseline)

    odors_labels = ["Toluene", "Benzene", "Methane", "CO", "Ammonia", "Acetone", "Acetaldehyde", "Methanol", "Butanol", "Ethylene"]
    odors_raw = [Toluene_raw, Benzene_raw, Methane_raw, CO_raw, Ammonia_raw, Acetone_raw, Acetaldehyde_raw, Methanol_raw, Butanol_raw, Ethylene_raw]
    odors_baseline = [Toluene_baseline, Benzene_baseline, Methane_baseline, CO_baseline, Ammonia_baseline, Acetone_baseline, Acetaldehyde_baseline, Methanol_baseline, Butanol_baseline, Ethylene_baseline]
    
    odors_labels = [odors_labels[i] for i, sublist in enumerate(odors_raw) for item in sublist] # Flattening
    odors_raw = [item for sublist in odors_raw for item in sublist] # Flattening
    odors_baseline = [item for sublist in odors_baseline for item in sublist] # Flattening

    return odors_raw, odors_baseline, odors_labels


def offset_subtraction(odors_raw, odors_raw_baseline):
    """
    Subtract baseline from raw data, which allows a more truthful validation of classification accuracy

    :param list raw: list of np arrays of odour recordings
    :param list baseline: list of 1-d np arrays baseline recordings
    :return _type_: list of baseline subtracted odour recordings
    """
    normalized_odors_raw = []
    for odor_traces, baselines in zip(odors_raw, odors_raw_baseline):
        normalized_odors_raw.append(odor_traces-baselines)

    return normalized_odors_raw

def pad_dataset(odors_raw):
    """
    Pad with zeros so that every trace has same number of time steps

    """
    max_steps = np.max([len(d) for d in odors_raw])

    padded_odors_raw = np.zeros((len(odors_raw), max_steps, len(odors_raw[0][0]))).astype(int)
    for i, trace in enumerate(odors_raw):
        padded_odors_raw[i,:len(trace)] = trace

    return padded_odors_raw

def zero_before_baseline(odors_raw, time_baseline, output_dt, sample_dt=1e-3):
    """
    Set every trace to zero before the baseline measurement
    time_baseline : measurement time in [s]
    output_dt : downsampled time step width

    """
    downsample_factor = int(output_dt/sample_dt)

    for i, _ in enumerate(odors_raw):
        odors_raw[i][:int(time_baseline*1000//downsample_factor)] = 0
    return odors_raw


def run(dir_data, dir_pickle_files):
    
    # Define experiments
    all_experiments = {
        "experiment" : {
            "OFFSET_SUBTRACTION": True,
            "DATASET_SIZE" : 20,
            "PAD_DATASET" : True,
            "OUTPUT_DT": 0.005,
            "TIME_BASELINE": 1,
            "ZERO_BEFORE_BASELINE": True
        }, 
    }

    # Iterate over experiments
    for i, (experiment, params) in enumerate(all_experiments.items()):
        OFFSET_SUBTRACTION = params["OFFSET_SUBTRACTION"]
        DATASET_SIZE = params["DATASET_SIZE"]
        PAD_DATASET = params["PAD_DATASET"]
        TIME_BASELINE = params["TIME_BASELINE"]
        ZERO_BEFORE_BASELINE = params["ZERO_BEFORE_BASELINE"]

        OUTPUT_DT = params["OUTPUT_DT"]

        experiment_name = "dataset_size_" + str(DATASET_SIZE) + "_baseline_subtraction_" + str(OFFSET_SUBTRACTION)
        random.seed(1)

        print(f"------ Generating {experiment_name} ------")

        # Extract data used in paper
        odors_raw, odors_raw_baseline, odor_labels = loadData(data_dir=dir_data, n_samples=DATASET_SIZE, output_dt=OUTPUT_DT, time_baseline=TIME_BASELINE)

        # Subtract Offset
        if OFFSET_SUBTRACTION:
            odors_raw = offset_subtraction(odors_raw, odors_raw_baseline)

        # pad dataset so that all traces have same size
        if PAD_DATASET:
            odors_raw = pad_dataset(odors_raw)

        # set all traces to zero before baseline measurement
        if ZERO_BEFORE_BASELINE:
            odors_raw = zero_before_baseline(odors_raw, time_baseline=TIME_BASELINE, output_dt=OUTPUT_DT)


        wf = open(dir_pickle_files.joinpath(experiment_name + "_data.pkl"), 'wb')
        print(f"Saving dataset as {experiment_name}_data.pkl")
        pickle.dump(odors_raw, wf, protocol=2) 
        # pickle.dump(odors_raw_testing, wf, protocol=2)
        wf.close()

        wf = open(dir_pickle_files.joinpath(experiment_name + "_labels.pkl"), 'wb')
        print(f"Saving dataset as {experiment_name}_labels.pkl")
        pickle.dump(odor_labels, wf, protocol=2) 
        # pickle.dump(odors_raw_testing, wf, protocol=2)
        wf.close()


if __name__ == '__main__':
    dir_pickle_files = Path('./')
    dir_pickle_files.mkdir(exist_ok=True, parents=True)
    run(Path('data'), dir_pickle_files)
