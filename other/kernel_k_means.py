import sys
from pathlib import Path
ROOTDIR = Path(__file__).resolve().parents[1]
if str(ROOTDIR / 'utils') not in sys.path:
    sys.path.append(str(ROOTDIR / 'utils'))

from utils import * 
from math import exp 
from LoadData import * 
import os
import time
from sklearn.cluster import KMeans
import numpy as np


# We use RBF kernel here
def kernel(data, sigma=4):

    nData = len(data)
    # nData x nData matrix
    Gram = np.zeros((nData, nData))
    for i in range(nData):
        for j in range(i,nData):
            if i != j: 
                square_dist = squaredDistance(data[i],data[j])
                base = 2.0 * sigma**2
                Gram[i][j] = exp(-square_dist/base)
                Gram[j][i] = Gram[i][j]
    return Gram 

def kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_


if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit(1) 
    
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    savePath = sys.argv[3]

    data = loadPoints(trainFile) 
    label = loadClusters(testFile) 
    normalize(data)

    '''
    filename=os.path.basename(trainFile).split('.')[0]
    setsname=os.path.basename(os.path.dirname(trainFile))
    # get kernel
    sigma=100.0
    if setsname=='shape':
        sigma=4.0
    elif setsname=='dim':
        sigma=50.0
    elif setsname=='overlap':
        sigma=100000.0
    elif setsname=='unbalance':
        sigma=50000.0
    elif setsname=='birch':
        sigma=100000.0
    if sigma==100.0:
        print("-------------------------wrong----------------------------")
        print("%s %s %f"%(setsname,filename,sigma))
        print("-------------------------wrong----------------------------")
        exit(1)
    print("%s %s %f"%(setsname,filename,sigma))
    '''
    data = kernel(data)  

    start=time.clock()

    # decide the number of clusters
    K = len(set(label))
    print("number of clusters: ",K)

    # run K-means
    results = kmeans(data, K) 

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