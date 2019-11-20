import numpy as np
import random
from sklearn.cluster import KMeans

def randsrc(row, column, rang):
    return [[random.choice(rang) for _ in range(column)] for _ in range(row)]

def sqdist(a, b):
    aa = np.sum(np.square(a), axis = 1)
    bb = np.sum(np.square(b), axis = 1)
    ab = np.dot(a.T, b)
    return abs(np.tile(aa.T, (1, bb.shape[1])) + np.tile(bb.T, (aa.shape[1], 1)) - 2*ab)

def meanInd(X, label, c, Z):
    n = X.shape[1]
    if np.unique(label).size != c:
        label = KMeans(init='k-means++', n_clusters=c).fit(X.T).labels_
        Z = np.ones([n, c])
    means = np.zeros(shape = (X.shape[0], c))
    for i in range(c):
        sub_idx = np.where(label == i)[0]
        means[:, i] = np.dot(np.array([[X[k][j] for j in sub_idx] for k in range(X.shape[0])]), Z[sub_idx, i]) / np.sum(Z[sub_idx, i])
    return means

def kmm(X, c, m, k = -99999):
# Input:
#       - X: the data matrix of size nFea x nSmp, where each column is a sample
#               point
#       - c: the number of clusters
#       - m: the number of multiple means(MM)
#       - k: the number of neighbor points   
    Ah = []#numpy.matrix()
    laKMMh = [] #numpy.matrix()
    Iter = 15
    OBJ = 0

    if k == -99999:
        if m < 6:
            k = c - 1
        else:
            k = 5

    n = X.shape[1]
    m0 = m
    Success = 1
    method = 1

    if method == 0:
        idx = [random.choice(range(n)) for _ in range(m)] #randsrc(m, 1, range(n))   # 1 ~ n in Matlab, here should be 0 ~ n-1
        Dis = sqdist(X, np.array([[X[i][j] for j in idx] for i in range(X.shape[0])]))
        StartIndZ = np.argmin(Dis, axis = 1) # 1-d array
    else:
        StartIndZ = KMeans(init='k-means++', n_clusters=m).fit(X.T).labels_

    BiGraph = np.ones([n, m])
    A = meanInd(X, StartIndZ, m, BiGraph)

    Ah = np.hstack((Ah, A))
    
if __name__ == "__main__":
    # kmm(np.array([[1,2,3],[4,5,6],[7,8,9]]), 3, 4)
    X = np.array([[ 1. ,  2. ],
       [ 1. ,  1.5],
       [ 0. ,  1. ],
       [ 9. , 10. ],
       [ 9. ,  9.5],
       [10. ,  9.5]])
    label = KMeans(init = 'k-means++', n_clusters = 2).fit(X).labels_
    c = 2
    Z = np.ones([6, 2])
    print(meanInd(X.T, label, c, Z))