import numpy as np
from sqdist import sqdist
import scipy.sparse as ss
def ConstructA_NP(A, B, k = 5, isSparse = 1):
    n = A.shape[1]
    # print('n: ' + str(n))
    if B.size == 0 or n == B.shape[1]:
        B = A
        m = n
        if n > 10000:
            block_size = 10
            save_type = 3
            Dis = gen_nn_distanceA(A.T, k+1, block_size, save_type)
            distXt = Dis.copy()
            di = np.zeros([n, k+1])
            id1 = di.copy()
            for i in range(k+1):
                di[:,i] = np.max(distXt, axis = 1)
                id1[:,i] = np.argmax(distXt, axis = 1)
                temp = (id1[:, i]) * n + range(n)
                distXt[(temp % (temp.shape[0])).astype(int),(temp / (temp.shape[0])).astype(int)] = 1e100
            
            id1 = np.fliplr(id1)
            di = np.fliplr(di)
        else:
            Dis = sqdist(A, B)
            distXt = Dis.copy()
            di = np.zeros([n, k+2])
            id1 = di.copy()
            for i in range(k+2):
                di[:,i] = np.min(distXt, axis = 1)
                id1[:,i] = np.argmin(distXt, axis = 1)
                temp = (id1[:, i]) * n + range(n)
                distXt[(temp % (temp.shape[0])).astype(int),(temp / (temp.shape[0])).astype(int)] = 1e100

            di = np.delete(di, 0, 1)
            id1 = np.delete(id1, 0, 1)

    else:
        Dis = sqdist(A, B)
        # print("Dis: "+ str(Dis.shape))
        # np.savetxt('Dis.txt', Dis)
        distXt = Dis.copy()
        di = np.zeros([n,k+1])
        id1 = di.copy() 
        for i in range(k+1):
            # print(di[:,i].shape)
            # print(np.min(distXt, axis = 1).reshape((-1, 1)).shape)
            di[:,i] = np.min(distXt, axis = 1).reshape((-1))
            id1[:,i] = np.argmin(distXt, axis = 1).reshape((-1))
            temp = (id1[:, i]) * n + range(n)
            # distXt[[x%(distXt.shape[0]) for x in temp],[int(x/(distXt.shape[0])) for x in temp]] = 1e100
            # print('\ntemp: ')
            # print(temp)
            distXt[(temp % (temp.shape[0])).astype(int),(temp / (temp.shape[0])).astype(int)] = 1e100
    
    m = B.shape[1]
    id1 = np.delete(id1, -1, 1)
    # np.savetxt('di.txt', di)
    Alpha = 0.5 * (k * di[:, k].astype(np.float64) - np.sum(di[:, range(k)], axis = 1).astype(np.float64))
    # np.savetxt('firstpart.txt', di[:, k])
    # np.savetxt('secondpart.txt', di[:, range(k)])
    # np.savetxt('Alpha.txt', Alpha)
    # print(di.shape)
    # print(k)
    # print(di[:, k].reshape(di.shape[0], 1).shape)
    # print(di[:, range(k)].shape)
    # print(di[:, k].reshape(di.shape[0], 1) - di[:, range(k)])
    # print((2 * Alpha + np.finfo(float).eps).reshape((-1)))
    tmp = (di[:, k].reshape(di.shape[0], 1) - di[:, range(k)]) / (2 * Alpha + np.finfo(float).eps).reshape((-1, 1))
    # print('tmp: '+ str(tmp.shape))
    rr = np.tile(np.array(range(n)), (1,k)).reshape(-1)
    cc = id1.reshape(-1, order='F')
    Z = ss.csr_matrix((tmp.reshape(-1, order = 'F'), (rr,cc)), shape=(n,m))

    # print('Z: ' + str(Z.shape/))

    if not isSparse:
        Z = Z.todense()
    
    return Z, Alpha, Dis, id1, tmp
    