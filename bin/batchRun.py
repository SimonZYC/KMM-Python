#!/usr/bin/env python
import os
import sys
import multiprocessing
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]

# execNames =["other/k_means","other/kernel_k_means","other/spectral",'kmm/KMM']  
execNames =['kmm/KMM']  
datasets=['brich','dim','overlap','shape','unbalance']
    

def run(execName,datapath,groundpath,savepath):
    cmdLine = "cd %s && pipenv run python %s.py %s %s %s" % (str(ROOTDIR), str(ROOTDIR / execName), datapath,groundpath,savepath)
    print(cmdLine)
    os.system(cmdLine)

def batchRun(setFilePath,  desFilePath, dataSet, nprocess=12):
    pool = multiprocessing.Pool(processes = nprocess)

    for execName in execNames:  # enumerate all methods
        tmpFilePath= os.path.join(desFilePath, execName)
        if not os.path.exists(tmpFilePath):
            os.makedirs(tmpFilePath)

        for datasets in os.listdir(setFilePath): # enumerate different exps
            setspath=os.path.join(setFilePath,datasets)
            sets=set()
            for file in os.listdir(setspath):   # get unique datasets
                sets.add(file.split('.')[0])
            for file in sets:
                datapath=os.path.join(setspath,file+'.data')
                groundpath=os.path.join(setspath,file+'.ground')
                savepath=os.path.join(tmpFilePath,file)
                pool.apply_async(run,(execName,datapath,groundpath,savepath))
                      
    pool.close()
    pool.join()
    
def exp0(dataSet, nprocess=8):
    
    setFilePath = ROOTDIR / "data"
    desFilePath = ROOTDIR / "EXP"
    if not os.path.exists(str(desFilePath)):
        os.makedirs(str(desFilePath))
    batchRun(str(setFilePath), str(desFilePath), dataSet, nprocess)

    
if __name__ == "__main__":
    nprocess = 6
    exp0(nprocess)
    
