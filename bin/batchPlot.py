from itertools import cycle, islice
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import os
import sys
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]

def loadClusters(filename): 
    input = open(filename, "r") 
    clusters=[]
    while True:
        line = input.readline().strip() 
        if not line:
            break
        clusters.append(int(line))
    return clusters


def loadPoints(filename):
    input = open(filename, "r")
    info = input.readline().split()
    
    nData = int(info[0]) # number of instances
    nDim = int(info[1])  # number of features
    
    data = [[0]*nDim for i in range(nData)]

    for i in range(nData):
        info = input.readline().split()
        for j in range(nDim):
            data[i][j] = float(info[j]) 

    return np.array(data)


execNames =["other/k_means","other/kernel_k_means","other/spectral",'kmm/KMM']  
# execNames=['KMM']
datasets=['birch','overlap','shape','unbalance']
# datasets = ['happyface']

setFilePath = str(ROOTDIR / "data")
desFilePath = str(ROOTDIR / "EXP")


H_plot=1  # plot for every dataset now
W_plot=len(execNames)


for dsets in datasets:
    setspath=os.path.join(setFilePath,dsets)
    sets=set()
    for file in os.listdir(setspath):   # get unique datasets
        sets.add(file.split('.')[0])
    # lock a specific dataset
    for file in sets:

        # load data
        dataPath=os.path.join(setspath,file+'.data')
        X=loadPoints(dataPath)

        # figure size inch
        plt.figure(figsize=(W_plot * 2 + 4, H_plot * 2 + 1,))
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
  

        for index, execName  in enumerate(execNames):
            # load labels

            clusterPath=os.path.join(desFilePath,execName,file+'.label')
            if not os.path.exists(clusterPath):
                continue

            Y=loadClusters(clusterPath)
            colors = np.array(list(islice(cycle(seaborn.color_palette('hls', int(max(Y) + 1))), int(max(Y) + 1))))

    
            plt.subplot(H_plot, W_plot, index+ 1)  
            print('subplot ',clusterPath)
            plt.scatter(X[:, 0], X[:, 1], color=colors[Y], s=6, alpha=0.6)
            plt.title(execName)

        if not os.path.exists(os.path.join(desFilePath,'images')):
            os.makedirs(os.path.join(desFilePath,'images'))
        savePath=os.path.join(desFilePath,'images',file+'.png')
        print(savePath)
        plt.savefig(savePath)
