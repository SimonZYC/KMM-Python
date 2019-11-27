'''
@Author: your name
@Date: 2019-11-24 15:28:25
@LastEditTime: 2019-11-27 11:22:32
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /KMM-Python/bin/test_kmm.py
'''
import sys
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOTDIR / 'kmm'))

from KMM import kmm
import argparse
import numpy as np
import math

if __name__ == '__main__':
    # print(str(sys.argv))
    parser = argparse.ArgumentParser(description='KMM Test Scripts')
    parser.add_argument('-d', '--data', type=str, default=str(ROOTDIR / 'kmm' / 'data' / 'happyface.csv'), help='path to the source data')
    parser.add_argument('-g', '--ground', type=str, default=str(ROOTDIR / 'kmm' / 'data' / 'happyface.ground.csv'), help='path to the ground label')
    parser.add_argument('-o', '--output', required = False, help = 'sum the integers (default: find the max)')

    args = parser.parse_args()
    datapath = args.data if Path(args.data).is_absolute() else Path('.').joinpath(args.data).absolute()
    groundpath = args.ground if Path(args.ground).is_absolute() else Path('.').joinpath(args.ground).absolute()
    
    X = np.matrix(np.loadtxt(datapath, dtype=np.float))
    Y = np.array(np.loadtxt(groundpath, dtype=np.int))
    n = len(X)
    c = np.unique(Y).size
    print('c: ' + str(c))
    m = math.floor(math.sqrt(n*c))
    k = 5
    laKMM,_,_,A,_,Ah,laKMMh = kmm(X.T, c, m,k)
    np.savetxt('laKMM.txt', laKMM, fmt='%d')

    
    
    # laKMM,_,_,A,_,Ah,laKMMh = kmm(X.T, c, m,k)

    # print(args.i)
