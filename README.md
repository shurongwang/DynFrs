# DynFrs

This is the official implementation for DynFrs (Anonymous, 2024), which is an efficient framework to perform machine unlearning (training sample removal) and online learning (training sample insertion) in Random Forests.

## Installation

Just simply git clone this project or download it.

## Usage

First, compile `main.cpp` with
```
g++ main.cpp -o dynfrs -std=c++17 -O3
```
Then, execute `dynfrs` with the following flags:

| Flag                     | Description                                                                | Arguments                                                                                                                                                                   |
|--------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-data`                  | set which dataset to train on                                              | dataset names <br> (e.g., `Purchase`, `Vaccine`, `Adults`, etc.)                                                                                                            |
| `-auto`                  | set hyperparameters automatically (with supported datasets only)           | none                                                                                                                                                                        |
| `-T`                     | set the number of trees in the forest                                      | an integer $T$ denoting the number of trees in the forest <br> (e.g., `100`, `250`, etc.)                                                                                   |
| `-k`                     | set $k = \lceil qT \rceil$ so that each sample occurs in at most $k$ trees | an integer $k$ between `1` and $T$ denoting the sample occurrence in the forest <br> (e.g., `10`, `25`, etc.)                                                               |
| `-d`                     | set the maximum depth of each tree                                         | an integer $d$ denoting the maximum tree depth <br> (e.g., `20`, `30`, etc.)                                                                                                |
| `-s`                     | set the number of candidate splits for each attribute                      | an integer $s$ denoting the number of candidate splits for each attribute <br> (e.g., `5`, `20`, etc.)                                                                      |
| `-acc`                   | perform evaluation of accuarcy on testing samples                          | none                                                                                                                                                                        |
| `-auc`                   | perform evaluation on AUROC on testing samples                             | none                                                                                                                                                                        |
| <nobr>`-unl_time`</nobr> | count the number of samples unlearned until elapsed time reached $t$       | an integer $t$ denoting the allowed unlearning time <br> (e.g., `1000`, `5000`, etc.)                                                                                       |
| `-unl_cnt`               | measure the time used for unlearning $n$ samples                           | an integer $n$ denoting the number of samples to be unlearned <br> (e.g., `100`, `1000`, etc.)                                                                              |
| `-stream`                | run DynFrs on an online mixed data stream                                  | three integers denoting the number of <br> (1) sample addition request <br> (2) sample deletion request <br> (3) querying request <br> (e.g., `50 50 100`, `5 5 100`, etc.) |


For instance,
```
./dynfrs -data Adult -auto -unl_cnt 100 -acc
```
will train DynFrs on the Adult dataset, and automatically set the hyperparameters as listed in Appendix. Then it will report the time for unlearning 100 samples. After that, it evaluate the unlearned model's predictive accuracy.
```
./dynfrs -data Higgs -T 100 -k 10 -d 30 -s 20 -stream 500000 500000 1000000
```
will train DynFrs on the Higgs dataset with $T=100$, $k=10$ (i.e., $q=0.1$), $d=30$, $s=20$, and then feed it with a online mixed data stream with $5\times10^5$ sample addition requests, $5\times10^5$ sample deletion requests$, and $10^6$ querying requests. It will report the average, minimum, and maximum delay for each type of request, and the percentage of correct prediction.

## Datasets

All datasets except Synthetic and Higgs are included in the repo. For Synthetic and Higgs, run the following command to generate/download and preprocess the data: (replace `<dataset>` with `Synthetic` or `Higgs`)
```
cd Datasets/<dataset> ; sh gen.sh
```

For external datasets, create two files `train.txt` and `test.txt` under `Datasets/Other`, with the following format:
```
<number of rows> <number of columns>
X_11 X_12 ... X_1d Y_1
X_21 X_22 ... X_2d Y_2
...  ...  ... ...  ...
X_n1 X_n2 ... X_nd Y_n
```
where `<number of rows>` should equal $n$ denoting the number of samples, and `<number of columns>` should equal $d+1$ denoting the number of attributes of each samples plus one. Note that the label of each sample is placed after each sample within the same line.
