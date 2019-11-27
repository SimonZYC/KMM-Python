# KMM-Python
COMP 5331 group project

Reference: [Paper](Paper: https://dl.acm.org/citation.cfm?id=3330846),  [Matlab code](https://github.com/CHLWR/KDD2019_K-Multiple-Means)



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
### Sample

### Custom
```
cd <Path-to-KMM>

pipenv run python bin/test_kmm.py -d data/overlap/s4.data -g data/overlap/s4.ground

```
## Source Code Description

KMM-Python

- bin: 
  - test_KMM.py: test scripts
  - Batch_plot.py: create a figure of the results

​	

- kmm: KMM code directory:
  - kmm.py: main function, control the iteration process.
  - csbg.py: Solve the optimization problemm
  - ConstructA_NP.py: create sparse matrix of distance between points
  - gen_nn_distanceA.py: calculate distance between points
  - sqdist.py: computes a rectangular matrix of pairwise distance between points in A (given in columns) and points in B
  - SVD2UV.py: transformation
- other: Implementaion of k-means, kernel-k-means, spectral
- utils: some tool functions

​		