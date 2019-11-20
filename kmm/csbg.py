import math
import scipy.sparse as ss
import numpy as np
import gc
import pickle
import os,sys

from main import sqdist
from SVD2UV import svd2uv

def clear(*arg):
    for i in arg:
        del arg
    gc.collect()

def gen_nn_distanceA(data, num_neighbors, block_size, save_type):
    n = data.shape[0]
    num_iter = math.ceil(n/block_size)
    A = ss.lil_matrix((n, n))
    dataT = data.T

    tmp = np.sum(np.square(data), axis = 1)
    # TODO: to dense matrix
    aa = np.array([[tmp[k][j] for j in np.ones(block_size)] for k in range(tmp.shape[0])])
    clear(tmp)

    for i in range(num_iter):
        
        start_index = i * block_size    #1 + (i-1) * block_size
        end_index = min((i+1)*block_size, n)# min((i+1)*block_size-1, n)  #i*block_size
        num_data = end_index - start_index

        block = dataT[:, range(start_index, end_index)]

        if num_data < block_size:
            aa = aa[:, range(0, num_data)]

        tmp = np.sum(np.square(block), axis = 0)
        # TODO: to dense matrix
        bb = tmp[np.zeros(n, int),:]
        clear(tmp)
        ab = np.dot(data, block)
        dist = aa + bb - 2 * ab

        clear(bb, ab, block)
        dist[dist < 0] = 0

        value = np.sort(dist, axis = 0)
        index = np.argsort(dist, axis = 0)
        tempindex = index[1:num_neighbors+1, :]
        rowindex = tempindex.reshape(tempindex.shape[0]*tempindex.shape[1], 1)
        tempindex = np.tile(np.array(range(num_data)), [num_neighbors, 1])
        columnindex = tempindex.reshape(tempindex.shape[0]*tempindex.shape[1], 1)
        tempvalue = value[1:num_neighbors+1, :]
        value = tempvalue.reshape(tempvalue.shape[0]*tempvalue.shape[1], 1)
        value[value < 1.0e-12] = 1.0e-12
        value = np.sqrt(value)

        A[:, start_index:end_index] = ss.csc_matrix((value, (rowindex, columnindex)), shape = (n, num_data))

    # clear()
    A1 = ss.triu(A)
    A1 = A + A1.T
    A2 = ss.tril(A)
    A2 = A2 + A2.T
    clear(A)
    max_num = 100000
    if(n < max_num):
        # A = np.maximum(A1, A2)
        A = ss.lil_matrix(np.maximum(A1.todense(), A2.todense()))
    else:
        num_iter = math.ceil(n / max_num)
        B = ss.lil_matrix((max_num,max_num))
        for i in range(num_iter):
            start_index = i * num_iter    #1 + (i-1) * num_iter
            end_index = min((i+1)*num_iter, n)# min((i+1)*num_iter-1, n)  #i*num_iter
            B = ss.lil_matrix(np.max_num(A1[:, start_index:end_index].todense(), A2[:, start_index:end_index].todense()))
            
            with open('tmp_{}.pkl'.format(str(i)), 'wb') as f:
                pickle.dump(B, f)
            clear(B)
    
    clear(A1, A2)

    if n > max_num:
        A = ss.lil_matrix((n, n))
        for i in range(num_iter):
            with open('tmp_{}.pkl'.format(str(i)), 'rb') as f:
                A = ss.hstack((A, pickle.load(f)))
            os.remove('tmp_{}.pkl'.format(str(i)))

    n = A.shape[0]
    B = ss.spdiags(A.diagonal(), 0, n, n)
    A = A - B
    
    # TODO savetype.

    return A


def ConstructA_NP(A, B, k = 5, isSparse = 1):
    n = A.shape[1]
    if B.size == 0 or n == B.shape[1]:
        B = A
        m = n
        if n > 10000:
            block_size = 10
            save_type = 3
            Dis = gen_nn_distanceA(A.T, k+1, block_size, save_type)
            distXt = Dis
            di = np.zeros([n, k+1])
            id = di
            for i in range(k+1):
                di[:,i] = np.max(distXt, axis = 1)
                id[:,i] = np.argmax(distXt, axis = 1)
                temp = (id[:, i]) * n + range(n)
                distXt[[x%distXt.shape[0] for x in temp],[int(x/distXt.shape[0]) for x in temp]] = 0
            
            id = np.fliplr(id)
            di = np.fliplr(di)
        else:
            Dis = sqdist(A, B)
            distXt = Dis
            di = np.zeros([n, k+2])
            id = di
            for i in range(k+2):
                di[:,i] = np.min(distXt, axis = 1)
                id[:,i] = np.argmin(distXt, axis = 1)
                temp = (id[:, i]) * n + range(n)
                distXt[[x%distXt.shape[0] for x in temp],[int(x/distXt.shape[0]) for x in temp]] = 1e100

            di = np.delete(di, 0, 1)
            id = np.delete(id, 0, 1)

    else:
        Dis = sqdist(A, B)
        distXt = Dis
        di = np.zeros([n,k+1])
        id = di 
        for i in range(k+1):
            di[:,i] = np.min(distXt, axis = 1)
            id[:,i] = np.argmin(distXt, axis = 1)
            temp = (id[:, i]) * n + range(n)
            distXt[[x%distXt.shape[0] for x in temp],[int(x/distXt.shape[0]) for x in temp]] = 1e100
    
    m = B.shape[1]
    id = np.delete(id, -1, 1)
    Alpha = 0.5 * (k * di[:, k] - np.sum(di[:, range(k)], axis = 1))
    tmp = (di[:, k].reshape(di.shape[1], 1) - di[:, range()]) / (2 * Alpha + np.finfo(float).eps)
    rr = np.tile(np.array(range(n)), (1,k)).reshape(-1)
    cc = id.reshape(-1, order='F')
    Z = ss.csr_matrix((tmp.reshape(-1, order = 'F'), (rr,cc)), shape=(n,m))

    if not isSparse:
        Z = Z.todense()
    
    return Z, Alpha, Dis, id, tmp
    
def EProjSimplex_new(v, k=-1):
    if k == -1:
        k = 1
    ft = 1
    n = max(v.shape)

    v0 = v - np.mean(v) + k/n
    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1>0
            npos = np.sum(posidx, axis=0)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if ft > 100:
                x = max(v1, 0)
                break
        
        x = max(v1, 0)
    else:
        x = v0
    
    return x, ft

def struG2la(Z):
    n,m = Z.shape
    SS0 = ss.csr_matrix((n+m, n+m))
    SS0[:n, n:] = Z
    SS0[n:, :n] = Z.T
    clusternum, label = ss.csgraph.connected_components(SS0)
    label = label[:n]
    # Compare: Here is 1*n array not n*1 in the matlab program

    return clusternum, label

def loss(distX,Z,alpha,lambda,U,V ):
    n = Z.shape[0]
    m = Z.shape[1]
    a1 = np.sum(Z, axis = 1)
    D1a = 
    pass

def CSBG(X, c, A, k = -10, alpha = -1000, llambda = -1000):
# Input:
#       - X: the data matrix of size nFea x nSmp, where each column is a sample
#               point
#       - c: the number of clusters
#       - A: the matrix of multiple means(MM) of size nFea x nMM
#       - k: the number of neighbor points
# Output:
#       - laKMM: the cluster assignment for each point
#       - laMM: the sub-cluster assignment for each point
#       - BiGraph: the matrix of size nSmp x nMM
# Requre:
# 		ConstructA_NP.m
# 		EProjSimplex_new.m
# 		svd2uv.m
# 		struG2la.m
# Usage:
#       % X: d*n
#       [laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);
# Reference:
#
#	Feiping Nie, Cheng-Long Wang, Xuelong Li, "K-Multiple-Means: A Multiple-Means 
#   Clustering Method with Specified K Clusters," In The 25th ACM SIGKDD Conference
#   on Knowledge Discovery and Data Mining (KDD ¡¯19), August 4¨C8, 2019, Anchorage, AK, USA.
#
#   version 1.0 --May./2019 
#
#   Written by Cheng-Long Wang (ch.l.w.reason AT gmail.com)
    NITER = 30
    zr = 10e-5

    if k == -10:
        k = 5
    n = X.shape[1]
    m = A.shape[1]
    Z, Alpha, distX, id =  ConstructA_NP(X, A,k) # A is sparse
    ZT, AlphaT, distXT, idT =  ConstructA_NP(A,X,k)

    if alpha == -1000:
        alpha =  1*np.mean(Alpha, axis = 1)
        alphaT = 1*np.mean(AlphaT, axis = 1)
    
    if llambda == -1000:
        llambda = (alpha + alphaT) / 2

    Z0 = (Z + ZT.T)/2

    BiGraph, U, V, evc, _, _ = svd2uv(Z0, c)
    if np.sum(evc.reshape(-1, order = 'F')[:c]) > c*(1-zr):
        sys.exit('The original graph has more than {} connected component£¬ Please set k larger'.format(c))
    D1 = 1
    D2 = 1
    Ater = 0
    dxi = np.zeros((n,k))
    for i in range(n):
        dxi[i,:] =  distX[i,id[i,:]]
    dxiT = np.zeros((m,k))
    for i in range(m):
        dxiT[i,:] = distXT[i,idT[i,:]]

    OBJ=[]
    Ater=0
    for iter in range(NITER):
        U1 = D1*U
        V1 = D2*V
        dist = sqdist(U1.T, V1.T)
        tmp1 = np.zeros((n,k))
        for i in range(n):
            dfi = dist[i, id[i,:]]
            ad = -(dxi[i,:] + llambda*dfi) / (2*alpha)
            tmp1[i, :], _ =  EProjSimplex_new(ad)

        rr = np.tile(np.array(range(n)), (1,k)).reshape(-1)
        cc = id.reshape(-1, order='F')
        Z = ss.csr_matrix((tmp1.reshape(-1, order = 'F'), (rr,cc)), shape=(n,m))

        tmp2 = np.zeros((m,k))
        for i in range(m):
            dfiT = dist[idT[i,:], i]
            ad = (dxiT[i,:]-0.5*llambda*dfiT.T)/(2*alphaT)
            tmp2[i,:], _ = EProjSimplex_new(ad)
        
        rr = np.tile(np.array(range(m)), (1,k)).reshape(-1)
        cc = id.reshape(-1, order='F')
        ZT = ss.csr_matrix((tmp2.reshape(-1, order = 'F'), (rr,cc)), shape=(m,n))

        BiGraph = (Z + ZT.T) / 2
        U_old = U
        V_old = V
        BiGraph, U, V, evc, D1, D2 = svd2uv(BiGraph, c)

        fn1 = np.sum(evc.reshape(-1,  order = 'F')[:c])
        fn2 = np.sum(evc.reshape(-1,  order = 'F')[:(c+1)])

        if fn1 < c - zr:
            Ater = 0
            llambda = 2*llambda
        elif fn2 > c+1-zr:
            Ater = 0
            llambda = llambda / 2
        else:
            Ater += 1
            if Ater == 2:
                break
    
    print('csbg loop: ' + str(iter))

    laMM = id[:,0]

    clusternum, laKMM = struG2la(BiGraph)
    if clusternum != c:
        print('Can not find the correct cluster number: '+ c)

    isCov = (Ater == 2)
    
    

if __name__ == "__main__":
    pass