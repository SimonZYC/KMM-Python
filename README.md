# KMM-Python
COMP 5331 group project

Reference: [Paper](Paper: https://dl.acm.org/citation.cfm?id=3330846) ,  [Matlab code](https://github.com/CHLWR/KDD2019_K-Multiple-Means)



## Prerequisite

- OS: Linux & Mac OS(tested)

- Python 3.7.4

- pipenv

## Setup

```
cd <work-dir>
git clone https://github.com/SimonZYC/KMM-Python.git
cd KMM-Python

pipenv install
```

## Run
### Toy Sample (For demonstration)

```
cd <Path-to-KMM>
./bin/test.sh toy
```

### Test All (very slow)

```
cd <Path-to-KMM>
./bin/test.sh all
```

### Custom
```
cd <Path-to-KMM>

pipenv run python bin/test_kmm.py -d data/overlap/s4.data -g data/overlap/s4.ground

```
## Source Code Description

KMM-Python

- bin: 
  - test.sh, test_KMM.py: test scripts
  - batchPlot.py: create a figure of the results
  - batchRun.py: execute serveral clustering algorithms together

​	

- KMM: KMM code directory:
  - KMM.py: main function, control the iteration process.
  - csbg.py: Solve the optimization problemm
  - ConstructA_NP.py: create sparse matrix of distance between points
  - gen_nn_distanceA.py: calculate distance between points
  - sqdist.py: computes a rectangular matrix of pairwise distance between points in A (given in columns) and points in B
  - SVD2UV.py: transformation

- other: Implementaion of k-means, kernel-k-means, spectral:
  - k_means.py: the code of k-means algorithm
  - kernel_k_means.py: the code of kernel k-means algorithm
  - spectral.py: the code of spectral clustering algorithm

- utils: some tool functions
  - affinity.py: the affinity function and pruning function used in the spectral clustering algorithm
  - analysis.py:
  - kernels.py: the kernel functions used in the kernel k-means algorithm
  - LoadData.py: the functions used to load data points and related labels
  - utils.py: the functions used to calculate NMI, purity and distance

​		