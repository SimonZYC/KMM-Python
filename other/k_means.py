import sys
from pathlib import Path
ROOTDIR = Path(__file__).resolve().parents[1]
if str(ROOTDIR / 'utils') not in sys.path:
    sys.path.append(str(ROOTDIR / 'utils'))

from utils import * 
from LoadData import * 
import time
import random
import os


def computeSSE(data, centers, clusterID):
    sse = 0 
    nData = len(data) 
    for i in range(nData):
        c = clusterID[i]
        sse += squaredDistance(data[i], centers[c]) 
        
    return sse 

def updateClusterID(data, centers):

    nData = len(data) 
    nCenters = len(centers) 
    clusterID = [0] * nData
    dis_Centers = [0] * nCenters 
    for i in range(nData):
        for c in range(nCenters):
            dis_Centers[c] = squaredDistance(data[i], centers[c])
        # clusterID[i]: the id of the closest center (id ranges from 0 to K-1)
        clusterID[i] = dis_Centers.index(min(dis_Centers))
    return clusterID

def updateCenters(data, clusterID, K):
    nDim = len(data[0]) 
    centers = [[0] * nDim for i in range(K)] 

    # unique clusterID
    ids = set(clusterID)
    for id in ids:
        # i: index; j: element
        # indices of points that belong to cluster i
        indices = [i for i, j in enumerate(clusterID) if j == id]
        cluster = [data[i] for i in indices]
        if len(cluster) == 0:
            centers[id] = [0] * nDim
        else:
            # compute the centroids zip(* ) returns the numbers of each dimension
            centers[id] = [float(sum(col))/len(col) for col in zip(*cluster)]
    return centers 

def kmeans(data, K, maxIter = 100, tol = 1e-6):


    nData = len(data) 
    nDim = len(data[0]) 
    clusterID =[]
    lastDistance = 1e100
    
    # randomly intialize the clusters (radnomly choose some points)
    random.seed(int(time.time()))
    seq=random.sample(range(len(data)),K)
    centers = []
    for i in seq:
        centers.append(data[i])
    
    for iter in range(maxIter):
        clusterID = updateClusterID(data, centers) 
        centers = updateCenters(data, clusterID, K)
        curDistance = computeSSE(data, centers, clusterID) 
        # stopping criterion
        if lastDistance - curDistance < tol or (lastDistance - curDistance)/lastDistance < tol:
            print ("# of iterations:", iter )
            print ("last distance = ", lastDistance)
            print ("SSE = ", curDistance)
            return clusterID
        lastDistance = curDistance
        
    print ("# of iterations:", iter )
    print ("last distance = ", lastDistance)
    print ("SSE = ", curDistance)
    return clusterID



if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit(1) 
    
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    savePath = sys.argv[3]
    
    print('algorithm: ',sys.argv[0])
    print("input data file: ",trainFile)
    data = loadPoints(trainFile) 
    label = loadClusters(testFile) 
   
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


    