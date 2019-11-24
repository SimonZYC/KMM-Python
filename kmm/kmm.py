import numpy as np
import random
import math
from sklearn.cluster import KMeans
from csbg import ConstructA_NP
from csbg import CSBG
from sqdist import sqdist

def randsrc(row, column, rang):
    return [[random.choice(rang) for _ in range(column)] for _ in range(row)]

# def sqdist(a, b):
#     aa = np.sum(np.square(a), axis = 1)
#     bb = np.sum(np.square(b), axis = 1)
#     # print(aa)
#     # print(bb)
#     ab = np.dot(a.T, b)
#     return abs(np.tile(aa.T, (1, bb.shape[1])) + np.tile(bb, (aa.shape[1], 1)) - 2*ab)

def meanInd(X, label, c, Z):
    n = X.shape[1]
    if np.unique(label).size != c:
        label = KMeans(init='k-means++', n_clusters=c).fit(X.T).labels_
        Z = np.ones([n, c])
    means = np.zeros(shape = (X.shape[0], c))
    # print('label :' + str(label.shape))
    # print('n: ' + str(n))
    # print('X: ' + str(X.shape))
    # print('Z: ' + str(Z.shape))
    for i in range(c):
        sub_idx = np.where(label == i)[0]
        # print(i)
        # print(sub_idx)
        # means[:, i] = np.dot(np.array([[X[k][j] for j in sub_idx] for k in range(X.shape[0])]), Z[sub_idx, i]) / np.sum(Z[sub_idx, i])
        means[:, i] = X[:, sub_idx].dot(Z[sub_idx, i]) / np.sum(Z[sub_idx, i])
    
    # print('means: '+ str(means.shape))
    # print(means)
    return means

def kmm(X, c, m, k = -99999):
# [laKMM, laMM, BiGraph, Anc, ~, ~, ~]= KMM(X', c, m,k) : K-Multiple-Means
# Input:
#       - X: the data matrix of size nFea x nSmp, where each column is a sample
#               point
#       - c: the number of clusters
#       - m: the number of multiple means(MM)
#       - k: the number of neighbor points
# Output:
#       - laKMM: the cluster assignment for each point
#       - laMM: the sub-cluster assignment for each point
#       - BiGraph: the matrix of size nSmp x nMM
#       - A: the multiple means matrix of size nFea x nMM
#       - laKMMh: the history of cluster assignment for each point
# Usage:
#       % X: d*n
#       laKMM, laMM, AnchorGraph, Anchors, _, _, _= KMM(X', c, m,k) 
 
    Ah = np.matrix([])#numpy.matrix()
    laKMMh = np.matrix([]) #numpy.matrix()
    Iter = 15
    # OBJ = 0
    OBJ = []

    if k == -99999:
        if m < 6:
            k = c - 1
        else:
            k = 5

    n = X.shape[1]
    # print(n)
    # print(m)
    m0 = m
    Success = 1
    method = 1

    if method == 0:
        idx = [random.choice(range(n)) for _ in range(m)] #randsrc(m, 1, range(n))   # 1 ~ n in Matlab, here should be 0 ~ n-1
        Dis = sqdist(X, np.matrix([[X[i][j] for j in idx] for i in range(X.shape[0])]))
        StartIndZ = np.argmin(Dis, axis = 1) # 1-d array
    else:
        StartIndZ = KMeans(init='k-means++', n_clusters=m).fit(X.T).labels_
        # print(StartIndZ) # Different

    BiGraph = np.ones([n, m])
    A = meanInd(X, StartIndZ, m, BiGraph)

    Ah = np.hstack((Ah, A)) if Ah.size else A
    # print('Ah: ' + str(Ah.shape))
    # print(Ah)
    laKMM, laMM, BiGraph, isCov, obj, _, _ = CSBG(X, c, A, k)
    laKMMh=np.hstack((laKMMh, laKMM)) if laKMMh.size else laKMM
    # OBJ[0] = obj[-1] if hasattr(obj, "__getitem__") else obj
    # if hasattr(obj, "__getitem__"):
    #     print(type(obj))
    #     print(obj)
    #     OBJ.append(obj[-1])
    # else:
    #     OBJ.append(obj)
    OBJ.append(obj)

    iter1 = 1
    while iter1 < Iter:
        iter1 += 1
        if isCov:
            print('iter: '+str(iter1))
            # if hasattr(obj, "__getitem__"):
            #     OBJ.append(obj[-1])
            # else:
            #     OBJ.append(obj)
            OBJ.append(obj)
            # OBJ[0] = obj[-1] if hasattr(obj, "__getitem__") else obj
            if np.all(StartIndZ==laMM):
                print('all mid=end')
                return laKMM, laMM, BiGraph, A, OBJ, Ah, laKMMh
            elif np.unique(laMM).size != m:
                print('length(unique(EndIndZ))~=m')
                StartIndZ = laMM
                while np.unique(StartIndZ).size != m:
                    print('len mid != m')
                    A = A[:, np.unique(StartIndZ)]
                    m = np.unique(StartIndZ).size
                    if np.unique(StartIndZ).size > c:
                        BiGraph, _, _, id, _ = ConstructA_NP(X, A, k)
                        StartIndZ = id[:, 0]
                    else:
                        m = m0
                        StartIndZ = KMeans(init='k-means++', n_clusters=m).fit(X.T).labels_
                        BiGraph = np.ones((n,m))
                        A = meanInd(X, StartIndZ, m, BiGraph)
                        Success = 0

                if Success == 0:
                    Ah = np.array([])
                
                Ah = np.hstack((Ah, A)) if Ah.size else A
                Success = 1

            else:
                print('mid ~=end & len min=m')
                StartIndZ = laMM
                A = meanInd(X, StartIndZ, m, BiGraph)
                Ah = np.hstack((Ah, A)) if Ah.size else A
        else:
            print('0~=isCov')
            StartIndZ = KMeans(init='k-means++', n_clusters=m).fit(X.T).labels_
            A = meanInd(X, StartIndZ, m, BiGraph)
            Ah = np.array([])
            Ah = np.hstack((Ah, A)) if Ah.size else A

        laKMM, laMM, BiGraph, isCov, obj, _, _ = CSBG(X, c, A, k)   
        laKMMh=np.hstack((laKMMh, laKMM)) if laKMMh.size else laKMM
    
    print('loop: '+ str(iter1))

    return laKMM, laMM, BiGraph, A, OBJ, Ah, laKMMh

if __name__ == "__main__":

    X = np.matrix(np.loadtxt('kmm/dat/X.csv', dtype=np.float, delimiter="\t"))
    # X = np.matrix(np.loadtxt('faceM.csv', dtype=np.float, delimiter=","))
    n = 1000
    c = 4
    m = math.floor(math.sqrt(n*c))
    k = 5
    laKMM,_,_,A,_,Ah,laKMMh = kmm(X.T, c, m,k)
    np.savetxt('laKMM.txt', laKMM, fmt='%d')
    # np.savetxt('laKMMh.txt', laKMMh, fmt='%d')
