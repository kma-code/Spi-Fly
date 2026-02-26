import numpy as np
import copy
import json
import argparse

# import labels and receptor response data from DoOR 2.0.1
receptor_labels = np.loadtxt("DoOR_datasets/receptor_labels.txt", dtype='str', delimiter=',')
odorant_labels = np.genfromtxt("DoOR_datasets/odorant_labels.txt", dtype='str', delimiter='\n')
# # For the DoOR datasets, the odorant labels are also provided by their InChIKey
# odorant_labels = np.loadtxt("DoOR_datasets/odorant_labels_InChIKey.txt", dtype='str', delimiter=',')
receptor_responses = np.loadtxt("DoOR_datasets/receptor_responses.csv", delimiter=',')
DoOR_DATASET = True

# exclude all receptors which are not Or
Or_receptor_idx = np.where(np.char.find(receptor_labels, 'Or') == 0)[0]
receptor_labels = receptor_labels[Or_receptor_idx]
receptor_responses = receptor_responses[:,Or_receptor_idx]


def parse_experiment_arguments():
    """
        Parse the arguments for the test and train experiments
    """

    parser = argparse.ArgumentParser(description='Perform an odor space analysis, \
        reducing number of OR types to desired N_OR.')
    parser.add_argument('--params', type=str,
                        help='Path to the parameter .json-file containing\
                        list of names of odorants and N_OR.')
    parser.add_argument('--list', action='store_true', default=False,
                        help='List all available odorants/OR types')
    args = parser.parse_args()

    return args

# define separability of response matrix
def separability(response_mat):
    sep = 0.0
    # for each pair of rows (= odors)
    for i in range(response_mat.shape[0]):
        for j in range(response_mat.shape[0]):
            if i == j:
                continue
            # calculate distance between odor responses
            sep += np.linalg.norm(response_mat[i] - response_mat[j])
    return sep

# define iterative reduction algorithm
def reduce_ORs(data, n_rec_max):

    # take full data matrix (selected odors x all receptors)
    reduced_data = data.copy()
    selected_receptor_idx = np.arange(data.shape[1])
    
    # if n_rec_max is already same as number of ORs (before reduction)
    if len(selected_receptor_idx) == n_rec_max:
        return reduced_data, selected_receptor_idx

    while len(selected_receptor_idx) != n_rec_max:
        # reduce data matrix by one receptor and calculate distance
        sep_arr = []
        for i, rec_id in enumerate(selected_receptor_idx):
            # kill one receptor
            temp_reduced_data = reduced_data.copy()
            temp_reduced_data[:, i] = 0
            # calculate separability of newly reduced data
            sep_arr.append(separability(temp_reduced_data))

        # eliminate receptor that leaves the dataset with the highest separability
        sep_arr = np.array(sep_arr)
        selected_receptor_idx = np.delete(selected_receptor_idx, np.argmax(sep_arr))
        reduced_data = np.delete(reduced_data, np.argmax(sep_arr), axis=1)
    
    return reduced_data, selected_receptor_idx


if __name__ == '__main__':

    args = parse_experiment_arguments()

    if args.list:
        if DoOR_DATASET:
            print("List of all available odorants:")
            for i, lab in enumerate(odorant_labels):
                print(f"{str(i)}: {lab}")
        exit()


    # load param file
    if args.params:
        with open(args.params, 'r+') as file:
            print(f"Loading {args.params}")
            params = json.load(file)
            od_names = params["odorant_names"]
            N_OR = params["N_OR"]
    else:
        raise ValueError("No params file given, run odor_space_analysis.py --params 'params'.json")

    # convert odorant names into idx
    try:
        od_idx = np.array([np.where(odorant_labels == od_name) for od_name in od_names]).flatten()
    except:
        raise ValueError("Odorant name not recognized, check odorant_labels.txt for available odorants")

    assert DoOR_DATASET
    print(f"Loaded DoOR 2.0.1 dataset")
    print(f"Performing analysis for odorants with idx {od_idx}:")
    for od_id in od_idx:
        print(str(odorant_labels[od_id]))

    reduced_data, selected_receptor_idx = reduce_ORs(receptor_responses[od_idx], N_OR)

    print("\nBest receptors for N_OR =", N_OR, ":")
    print("Selected receptor idx:", selected_receptor_idx)
    print("Selected receptors:", receptor_labels[selected_receptor_idx])

    
    # save the odorants and OR into the params file for use with generate_dataset.py
    params["odorant_idx"] = [int(x) for x in od_idx]
    params["OR_idx"] = [int(x) for x in selected_receptor_idx]
    params["OR_names"] = list(receptor_labels[selected_receptor_idx])

    # save corresponding dict
    print(f"Saving results to {args.params}")
    with open(args.params, 'w') as f:
        json.dump(params, f, indent=2)