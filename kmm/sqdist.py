'''
@Author: your name
@Date: 2019-11-21 17:35:21
@LastEditTime: 2019-11-24 18:55:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /KMM-Python/kmm/sqdist.py
'''
import numpy as np

def sqdist(a, b):
    # print(a)
    # print(b)
    aa = np.matrix(np.sum(np.square(a), axis = 0))
    bb = np.matrix(np.sum(np.square(b), axis = 0))
    # print("aa: "+ str(aa.shape))
    # f = open('aa.txt', 'a')
    # np.savetxt(f, aa, fmt='%.4f', delimiter=',')
    # f.close()

    # print("bb: "+ str(bb.shape))
    # f = open('bb.txt', 'a')
    # np.savetxt(f, bb, fmt='%.4f', delimiter=',')
    # f.close()
    
    # ab = np.dot(a.T, b)
    ab = a.T * b
    # print("ab: "+ str(bb.shape))
    # f = open('ab.txt', 'a')
    # np.savetxt(f, ab, fmt='%.4f', delimiter=',')
    # f.close()
    # print(ab)
    # print(abs(np.tile(aa.T, (1, bb.shape[1])) + np.tile(bb, (aa.shape[1], 1)) - 2*ab))
    return abs(np.tile(aa.T, (1, bb.shape[1])) + np.tile(bb, (aa.shape[1], 1)) - 2*ab)