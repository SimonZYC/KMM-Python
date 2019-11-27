import sys
from pathlib import Path
ROOTDIR = Path(__file__).resolve().parents[1]
if str(ROOTDIR / 'utils') not in sys.path:
    sys.path.append(str(ROOTDIR / 'utils'))

from utils import * 
from LoadData import * 
import affinity
import numpy
import scipy
import time
from sklearn.cluster import KMeans
import os
import copy


def laplacian(A):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}
    """
    D = numpy.zeros(A.shape)
    w = numpy.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
    return D.dot(A).dot(D)


def kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_

def spectral_clustering(affinity, n_clusters):
    L = laplacian(affinity)                    # change the tol value here
    eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters,tol=0.001)
    X = eig_vect.real
    rows_norm = numpy.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = kmeans(Y, n_clusters)      # modified here
    return labels


if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit(1) 
    
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    savePath = sys.argv[3]
    
    data = loadPoints(trainFile) 
    label = loadClusters(testFile) 
    normalize(data)

    filename=os.path.basename(trainFile).split('.')[0]
    setsname=os.path.basename(os.path.dirname(trainFile))
    

    start=time.clock()

    # decide the number of clusters
    K = len(set(label))
    print("number of clusters: ",K)
    data =numpy.array(data)
    A = affinity.compute_affinity(data)     # modified here

    # run spectral clustering
    results = spectral_clustering(A, K) 
    end=time.clock()

    #evaluation
    res_Purity = purity(label, results) 
    res_NMI = NMI(label, results) 
    
    print ("Purity =", res_Purity)
    print ("NMI = ", res_NMI)
    print("time = ",end-start)
    print()

    with open(savePath,'a+') as f:
        f.write("Purity=%f\n"%res_Purity)
        f.write('NMI=%f\n'%res_NMI)
        f.write('time=%f\n'%(end-start))
    f.close()

    labelFile=savePath+'.label'
    if not os.path.exists(labelFile):
        with open(labelFile,'w') as f:
            for id in results:
                f.write('%s\n'%id)

        f.close()