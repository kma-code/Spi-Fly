# Gas sensor original dataset: drift compensation

Here, we generate and normalize the gas sensor dataset.

The folder `data` contains all sensor recordings used in the gas sensor dataset for this work.
The original data has been downloaded from [1].
Each file is one sample recorded by 72 sensors (9 boards * 8 sensors) of the dataset. There are 20 different recordings for each of the 10 compounds (we exclude CO_1000, because it has less recordings than the other 10 compounds).

Following [2], we take two steps to mitigate long- and short-term drift in the dataset:
- we mitigate long-term drift by subtracting the baseline after 1s
- we only use recordings at position/location 4 to mitigate short-term drift. Note that according to [1], the file ending in "p7" corresponds to location 4.

Run `python genData_extended.py` in this folder to generate the dataset.


[1] [https://archive.ics.uci.edu/dataset/251/gas+sensor+arrays+in+open+sampling+settings](https://archive.ics.uci.edu/dataset/251/gas+sensor+arrays+in+open+sampling+settings), Vergara et al.

[2] "Drift in a popular metal oxide sensor dataset reveals limitations for gas classification benchmarks", Dennler et al.
We build on the code provided by by Nik Dennler in [https://zenodo.org/records/6338624](https://zenodo.org/records/6338624).

