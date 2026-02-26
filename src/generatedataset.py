import os
import shutil
import subprocess
import sys
import dill
import json
import logging

def generate_dataset(dataset_id):

	# Directory where the dataset is saved
	saved_dataset_dir = f"saved_datasets/dataset{dataset_id}/"

	# # Path to the new directories for groups
	# group_dirs = {group: os.path.join(saved_datasets_dir, group) for group in groups}

	# Generate dataset and move into appropriate folder
	dataset_name = f"dataset_template_{dataset_id}.json"
	
	# if LOAD_DATASET:
	# 	logging.info(f"Loading dataset {dataset_name}")
	# else:
	logging.info(f"Generating new dataset. This may take a while")
	# Run the subprocess commands (assumed not to change)
	subprocess.run(["python", "odor_space_analysis.py", "--params", f"saved_datasets/dataset{dataset_id}/{dataset_name}"])
	subprocess.run([sys.executable, "generate_dataset.py", "--params", f"saved_datasets/dataset{dataset_id}/{dataset_name}"])

	# Load the dataset_dict after running subprocess
	with open(f"saved_datasets/dataset{dataset_id}/{dataset_name}") as f:
		dataset_dict = json.load(f)
	
	# Generate the filename for the generated voltage data
	voltage_file_name = f"dataset_size{dataset_dict['dataset_size']}_Nodor{dataset_dict['N_odorants']}_NOR{dataset_dict['N_OR']}_voltage.pkl"
	
	voltage_file_path_and_name = os.path.join(saved_dataset_dir, voltage_file_name)
		
	# Move the generated file to the appropriate subfolder
	if os.path.exists(saved_dataset_dir):
		shutil.move(f"saved_datasets/{voltage_file_name}", os.path.join(saved_dataset_dir, voltage_file_name))
		logging.info(f"Moved {voltage_file_name} to {saved_dataset_dir}")