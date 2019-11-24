import numpy as np

def sqdist(a, b):
    # print(a)
    # print(b)
    aa = np.matrix(np.sum(np.square(a), axis = 0))
    bb = np.matrix(np.sum(np.square(b), axis = 0))
    # print(aa)
    # print(bb)
    ab = np.dot(a.T, b)
    # print(ab)
    # print(abs(np.tile(aa.T, (1, bb.shape[1])) + np.tile(bb, (aa.shape[1], 1)) - 2*ab))
    return abs(np.tile(aa.T, (1, bb.shape[1])) + np.tile(bb, (aa.shape[1], 1)) - 2*ab)