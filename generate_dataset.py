import numpy as np
import copy
import json
import torch
import dill
import argparse

from synth_dataset import SynthDataset
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

# multiprocessing tools
import multiprocess as mp
import functools
import itertools

import time
import pathlib
CWD = pathlib.Path(__file__).parent

# import labels and receptor response data from DoOR 2.0.1
receptor_labels = np.loadtxt("DoOR_datasets/receptor_labels.txt", dtype='str', delimiter=',')
odorant_labels = np.genfromtxt("DoOR_datasets/odorant_labels.txt", dtype='str', delimiter='\n')
# # For the DoOR datasets, the odorant labels are also provided by their InChIKey
# odorant_labels = np.loadtxt("DoOR_datasets/odorant_labels_InChIKey.txt", dtype='str', delimiter=',')
receptor_responses = np.loadtxt("DoOR_datasets/receptor_responses.csv", delimiter=',')

# exclude all receptors which are not Or
Or_receptor_idx = np.where(np.char.find(receptor_labels, 'Or') == 0)[0]
receptor_labels = receptor_labels[Or_receptor_idx]
receptor_responses = receptor_responses[:,Or_receptor_idx]

def parse_experiment_arguments():
    """
        Parse the arguments for the test and train experiments
    """

    parser = argparse.ArgumentParser(description='Generate dataset by simulating ORN model.')
    parser.add_argument('--params', type=str,
                        help='Path to the parameter .json-file containing\
                        list of names of odorants and N_OR.')
    args = parser.parse_args()

    return args


"""
    Define global parameters below

"""

# simulation step width in s
# Note: for good matching of statistics with Butterwick et al,
# all simulations should be performed with dt = 5e-5:
dt = 5e-5

"""
    End of definitions

"""


if __name__ == '__main__':

    args = parse_experiment_arguments()

    # load param file
    if args.params:
        with open(args.params, 'r+') as file:
            print(f"Loading {args.params}")
            params = json.load(file)
            od_names = params["odorant_names"]
            N_OR = params["N_OR"]
            dataset_size = params["dataset_size"]
            output_dt = params["output_dt"]
            total_steps = params["total_steps"]
            data_steps = params["data_steps"]
            N_ORCOs = params["N_ORCOs"]
            odorant_idx = params["odorant_idx"]
            OR_idx = params["OR_idx"]
            output = params["output"]
    else:
        raise ValueError("No params file given, run generate_dataset.py --params 'params'.json")

    t0 = time.time()

    print("\nGenerating dataset using DoOR 2.0.1 database\n")
    print(f"dataset_size: {dataset_size}")
    print(f"output_dt: {output_dt}")
    print(f"downsampling factor: {int(output_dt/dt)}")
    print(f"N_ORCOs: {N_ORCOs}")

    for od_id, od_name in zip(odorant_idx, od_names):
        # make sure that we are using the right dataset
        assert odorant_labels[od_id] == od_name, "Odorant id does not match name.\
        Are you using the correct dataset?"
    
    print(f"odorant_idx: {odorant_idx}")
    print(f"odorant names: {od_names}")
    if isinstance(OR_idx, list):
        print(f"OR_idx: {OR_idx}")
        print(f"OR names: {receptor_labels[OR_idx]}")
    elif isinstance(OR_idx, str) and OR_idx == 'all':
        print(f"simulating all ORs")
        OR_idx = list(range(len(receptor_labels)))
    else:
        raise ValueError("OR_idx should be list or 'all'")
    print(f"output: {output}")
    print(f"total simulation length: {total_steps * dt}s")
    print(f"ligand binding simulation length: {data_steps * dt}s")

    # in order to simulate the traces in parallel, we run one subprocess per odorant
    partial_run = functools.partial(SynthDataset, OR_idx=OR_idx, dt=dt,
                                    total_steps=total_steps, data_steps=data_steps,
                                    dataset_size=dataset_size, N_ORCOs=N_ORCOs,
                                    output=output, output_dt=output_dt)

    N_PROCESSES = len(odorant_idx)
    # rng seeds
    seeds = np.arange(N_PROCESSES) + 472

    with mp.Pool(N_PROCESSES) as pool:
        print(f'\nSetting up and running {N_PROCESSES} processes\n')
        dataset = pool.starmap(partial_run, zip(seeds, [[x] for x in odorant_idx]))
        pool.close()

    # get dataset dict with simulation params and join to params file
    trainset_dict = copy.deepcopy(vars(dataset[0]))
    trainset_dict.pop('_SynthDataset__cs')
    trainset_dict.pop('_SynthDataset__vals')
    trainset_dict.pop('rng')
    trainset_dict.pop('odorant_idx')
    trainset_dict.pop('OR_idx')
    trainset_dict['N_odorants'] = len(odorant_idx)
    params = params | trainset_dict

    # combine datasets
    dataset = ConcatDataset(dataset)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    memory_size = 0
    for sample, target in train_loader:
        print("sample dimensions:", sample.size(), ", target id:", target, ", target name:", odorant_labels[target[0,1]])
        memory_size += sample[0].element_size() * sample[0].nelement() / 1024**2
        
    print("\nDataset size:", memory_size, "MByte")


    # Save the dataloader and attributes dict, so that later uses don't require recalculating all traces
    label = 'size' + str(params['dataset_size']) + '_Nodor' + str(params['N_odorants'])
    label += '_NOR' +str(len(params['OR_idx']))
    label += '_' + str(params['output'])
    # save dataloader
    print(f"Saving dataset to {'saved_datasets/dataset_' + label + '.pkl'}.")
    with open('saved_datasets/dataset_' + label + '.pkl', 'wb') as file:
        dill.dump(dataset, file)
    # save corresponding dict
    print(f"Saving dictionary to {args.params}")
    with open(args.params, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Total simulation time: {time.time() - t0} s")