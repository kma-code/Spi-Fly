# Code repository for Spi-Fly: Few-shot, continual learning for spiking neuromorphic olfaction

Code to reproduce all figures of "Few-shot, continual learning for spiking neuromorphic olfaction".

## Installation

To install, run:

```
python3 -m venv flyEnv
source flyEnv/bin/activate
pip3 install -r requirements.txt
python -m ipykernel install --user --name=flyEnv 
```


## Generating the datasets

Due to the large file size, the datasets are not uploaded here, but need to be generated from the raw data first.

### Synthetic DM dataset

Open `SNN_FlyModel.py` and make sure that `LOAD_DATASET = False`. Then, you can run any of the parameter files using the synthetic dataset, e.g.:

```
python SNN_FlyModel.py --params experiments/exp31_BPTT_full_multiseed/params0/seed0/params.yaml --no-neptune
```

Afterwards, you can set `LOAD_DATASET = True` to avoid generating the dataset at every run.

### Gas Sensor dataset

```
cd saved_datasets/gas_sensor_dataset
python genData_extended.py
```

## Running simulations

The individual runs can be started by running `python SNN_FlyModel.py --params experiments/params.yaml --no-neptune`.

Runs can be logged to neptune.ai. To do so, add your `api_token` to `neptune_id.json`. You can then leave out the flag `--no-neptune`.

`experiments/params.yaml` contains an example parameter file. All parameters are documented therein.

To reproduce the figures in the paper, run the following commands, then open `make_plots.ipynb`.

(There are 10 seeds for each experiment, which can be run by replacing `seed0` by `seed1` etc. in the folder names.)

### Figure 2: Full dataset

For the offline classifier results (Gaussian NB, SVM), take a look at `offline_classifiers.ipynb`.

#### Synthetic DM dataset

BPTT, best performance after 10 epochs:
`python SNN_FlyModel.py --params experiments/exp31_BPTT_full_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT, best performance after 1 epoch:
`python SNN_FlyModel.py --params experiments/exp32_BPTT_early_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly, best performance after 10 epochs:
`python SNN_FlyModel.py --params experiments/exp33_assoc_full_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly, best performance after 1 epoch:
`python SNN_FlyModel.py --params experiments/exp34_assoc_early_multiseed/params0/seed0/params.yaml --no-neptune`

#### Gas sensor DM dataset

BPTT, best performance after 10 epochs:
`python SNN_FlyModel.py --params experiments/exp40_gas_BPTT_full_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT, best performance after 1 epoch:
`python SNN_FlyModel.py --params experiments/exp39_gas_BPTT_early_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly, best performance after 10 epochs:
`python SNN_FlyModel.py --params experiments/exp42_gas_assoc_full_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly, best performance after 1 epoch:
`python SNN_FlyModel.py --params experiments/exp41_gas_assoc_early_multiseed/params0/seed0/params.yaml --no-neptune`


### Figure 3: Continual Learning

#### Synthetic DM dataset

BPTT:
`python SNN_FlyModel.py --params experiments/exp51_synth_CL_BPTT_scan/params0/seed0/params.yaml --no-neptune`

Spi-Fly:
`python SNN_FlyModel.py --params experiments/exp52_synth_CL_assoc_scan/params0/seed0/params.yaml --no-neptune`

Offline BPTT:
`python SNN_FlyModel.py --params experiments/exp60_synth_offlineCL_BPTT_reset_scan/params0/seed0/params.yaml --no-neptune`

EWC + BPTT:
`python SNN_FlyModel.py --params experiments/exp132_synth_EWC_multiseed/params0/seed0/params.yaml --no-neptune`


#### Gas sensor DM dataset

BPTT:
`python SNN_FlyModel.py --params experiments/exp126_gas_CL_BPTT_ADAMreset_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly:
`python SNN_FlyModel.py --params experiments/exp50_gas_CL_assoc_multiseed/params0/seed0/params.yaml --no-neptune`

Offline BPTT:
`python SNN_FlyModel.py --params experiments/exp127_gas_offlineCL_BPTT_reset_multiseed/params0/seed0/params.yaml --no-neptune`

EWC + BPTT:
`python SNN_FlyModel.py --params experiments/exp134_gas_EWC_multiseed/params0/seed0/params.yaml --no-neptune`




### Figure 4: Reduced bit precision

#### Synthetic DM dataset

BPTT 32 bit:
`python SNN_FlyModel.py --params experiments/exp31_BPTT_full_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT 8 bit:
`python SNN_FlyModel.py --params experiments/exp125_synth_BPTT_8bit_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT 6 bit:
`python SNN_FlyModel.py --params experiments/exp124_synth_BPTT_6bit_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT 4 bit:
`python SNN_FlyModel.py --params experiments/exp120_synth_BPTT_4bit_multiseed/params0/seed0/params.yaml --no-neptune`


Spi-Fly 32 bit:
`python SNN_FlyModel.py --params experiments/exp33_assoc_full_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly 8 bit:
`python SNN_FlyModel.py --params experiments/exp123_synth_assoc_8bit_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly 6 bit:
`python SNN_FlyModel.py --params experiments/exp122_synth_assoc_6bit_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly 4 bit:
`python SNN_FlyModel.py --params experiments/exp121_synth_assoc_4bit_multiseed/params0/seed0/params.yaml --no-neptune`


#### Gas sensor DM dataset

BPTT 32 bit:
`python SNN_FlyModel.py --params experiments/exp40_gas_BPTT_full_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT 8 bit:
`python SNN_FlyModel.py --params experiments/exp114_gas_BPTT_8bit_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT 6 bit:
`python SNN_FlyModel.py --params experiments/exp115_gas_BPTT_6bit_multiseed/params0/seed0/params.yaml --no-neptune`

BPTT 4 bit:
`python SNN_FlyModel.py --params experiments/exp112_gas_BPTT_multiseed/params0/seed0/params.yaml --no-neptune`


Spi-Fly 32 bit:
`python SNN_FlyModel.py --params experiments/exp42_gas_assoc_full_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly 8 bit:
`python SNN_FlyModel.py --params experiments/exp116_gas_assoc_8bit_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly 6 bit:
`python SNN_FlyModel.py --params experiments/exp117_gas_assoc_6bit_multiseed/params0/seed0/params.yaml --no-neptune`

Spi-Fly 4 bit:
`python SNN_FlyModel.py --params experiments/exp113_gas_assoc_multiseed/params0/seed0/params.yaml --no-neptune`


### Figure 5: Off-chip memory footprint

See the jupyter notebook `memory_estimate.ipynb`.