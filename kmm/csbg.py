import math
import scipy.sparse as ss
import numpy as np
import gc
import pickle
import os,sys

# from main import sqdist
from SVD2UV import svd2uv
from sqdist import sqdist
from ConstructA_NP import ConstructA_NP
from gen_nn_distanceA import gen_nn_distanceA


def EProjSimplex_new(v, k=-1):
    if k == -1:
        k = 1
    ft = 1
    n = max(v.shape)

    v0 = v - np.mean(v) + k/n
    # print('v0: ' + str(v0.shape))
    vmin = np.min(v0)
    # print(vmin)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1>0
            # npos = np.sum(posidx, axis=0)
            npos = np.sum(posidx)
            g = -npos
            # print(g)
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if ft > 100:
                x = np.maximum(v1, 0)
                break
        
        x = np.maximum(v1, 0)
    else:
        x = v0
    
    return x, ft

def struG2la(Z):
    n,m = Z.shape
    # print('Z: ' + str(Z.shape))

    SS0 = ss.csr_matrix((n+m, n+m))
    SS0[:n, n:] = Z
    SS0[n:, :n] = Z.T
    # np.savetxt('SS0.txt', SS0.todense())
    clusternum, label = ss.csgraph.connected_components(SS0.todense(), connection='strong')

    # print('cluster #: ' + str(clusternum))
    # np.savetxt('label.txt', label)
    label = label[:n]
    # Compare: Here is 1*n array not n*1 in the matlab program

    return clusternum, label

def loss(distX,Z,alpha,llambda,U,V ):
    n = Z.shape[0]
    m = Z.shape[1]
    a1 = np.sum(Z, axis = 1)
    D1a = ss.spdiags((1/np.sqrt(a1)).reshape(-1), 0, n, n)
    a2 = np.sum(Z, axis = 0)
    D2a = ss.spdiags((1/np.sqrt(a2.T)).reshape(-1), 0, m, m)
    st = np.sum(np.multiply(distX, Z))
    at = alpha * np.sum(np.power(Z, 2))
    Da = ss.spdiags(np.vstack((1/np.sqrt(a1), 1/np.sqrt(a2.T))).reshape(-1), 0, n+m, n+m)
    SS = ss.csr_matrix((n+m, n+m))
    SS[:n, n:] = Z
    SS[n:, :n] = Z.T
    ft = llambda * np.trace(np.vstack((U,V)).T.dot(np.eye(n+m) - Da.dot(SS).dot(Da)).dot(np.vstack((U,V))))
    obj = st+ at  + ft
    return obj

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
    # print('n: '+str(n)+' m: '+str(m))
    Z, Alpha, distX, id1, _ =  ConstructA_NP(X, A,k) # A is sparse
    ZT, AlphaT, distXT, idT, _ =  ConstructA_NP(A,X,k)
    # f1 = open('Z.txt', 'a')
    # np.savetxt(f1, Z.todense())
    # f1.write('\n')
    # f1.close()
    # f2 = open('ZT.txt', 'a')
    # np.savetxt(f2, ZT.todense())
    # f2.write('\n')
    # f2.close()
    # print('Z: ' + str(Z.shape))
    # print('Alpha: ' + str(Alpha.shape))
    # print('distX: ' + str(distX.shape))
    # print('id: ' + str(id1.shape))
    # print('ZT: ' + str(ZT.shape))
    # print('AlphaT: ' + str(AlphaT.shape))
    # print('distXT: ' + str(distXT.shape))
    # print('idT: ' + str(idT.shape))
    if alpha == -1000:
        # alpha =  1*np.mean(Alpha, axis = 1)
        # alphaT = 1*np.mean(AlphaT, axis = 1)
        alpha =  1*np.mean(Alpha)
        alphaT = 1*np.mean(AlphaT)
    
    # print('alpha: '+str(alpha))
    # print('alphaT: '+str(alphaT))

    if llambda == -1000:
        llambda = (alpha + alphaT) / 2

    Z0 = (Z + ZT.T)/2
    # print('Z0: ' + str(Z0.shape))
    # np.savetxt('Z0.txt', Z0.todense())
    BiGraph, U, V, evc, _, _ = svd2uv(Z0, c)
    # np.savetxt('BiGraph.txt', BiGraph)
    # np.savetxt('U.txt', U)
    # np.savetxt('V.txt', V)
    # np.savetxt('evc.txt', evc)

    if np.sum(evc.reshape(-1, order = 'F')[:c]) > c*(1-zr):
        sys.exit('The original graph has more than {} connected component£¬ Please set k larger'.format(c))
    D1 = 1
    D2 = 1
    Ater = 0
    dxi = np.zeros((n,k))
    for i in range(n):
        dxi[i,:] =  distX[i,id1[i,:].astype(int)]
    dxiT = np.zeros((m,k))
    for i in range(m):
        dxiT[i,:] = distXT[i,idT[i,:].astype(int)]
    # np.savetxt('dxi.txt', dxi)
    # np.savetxt('dxiT.txt', dxiT)

    OBJ=np.array([])
    Ater=0
    # print('U: ' + str(type(U)))
    # print('V: ' + str(type(V)))
    for iter1 in range(NITER):
        # print('D1: ' + str(type(D1)))
        # print('D2: ' + str(type(D2)))
        
        U1 = D1*U
        V1 = D2*V
        # print('U1: ' + str(type(U1)))
        # print('V1: ' + str(type(V1)))
        # if iter1 == 0 or iter1 == 1:
        #     f1 = open('U1.txt', 'a')
        #     np.savetxt(f1, U1)
        #     f1.write("\n\n")
        #     f2 = open('V1.txt', 'a')
        #     np.savetxt(f2, V1)
        #     f2.write("\n\n")
        #     f1.close()
        #     f2.close()
        # np.savetxt('U1.txt', U1)
        # np.savetxt('V1.txt', V1)

        dist = sqdist(U1.T, V1.T)
        tmp1 = np.zeros((n,k))
        for i in range(n):
            dfi = dist[i, id1[i,:].astype(int)]
            ad = -(dxi[i,:] + llambda*dfi) / (2*alpha)
            tmp1[i, :], _ =  EProjSimplex_new(ad)
        # if iter1==0:
        #     np.savetxt('tmp1.txt', tmp1)
        rr = np.tile(np.array(range(n)), (1,k)).reshape(-1)
        cc = id1.reshape(-1, order='F')
        # cc = id1.reshape(-1)
        Z = ss.csr_matrix((tmp1.reshape(-1, order = 'F'), (rr,cc)), shape=(n,m))
        # Z = ss.csr_matrix((tmp1.reshape(-1), (rr,cc)), shape=(n,m))
        # if iter1 == 0:
        #     np.savetxt('Z.txt', Z.todense())

        tmp2 = np.zeros((m,k))
        for i in range(m):
            dfiT = dist[idT[i,:].astype(int), i]
            ad = (dxiT[i,:]-0.5*llambda*dfiT.T)/(2*alphaT)
            tmp2[i,:], _ = EProjSimplex_new(ad)
        
        rr = np.tile(np.array(range(m)), (1,k)).reshape(-1)
        # print('rr: ' + str(rr.shape))
        cc = idT.reshape(-1, order='F')
        # cc = idT.reshape(-1)
        # print('cc: ' + str(cc.shape))
        # print('tmp2: ' + str(tmp2.shape))
        ZT = ss.csr_matrix((tmp2.reshape(-1, order = 'F'), (rr,cc)), shape=(m,n))
        # ZT = ss.csr_matrix((tmp2.reshape(-1), (rr,cc)), shape=(m,n))

        # if iter1 == 0:
        #     np.savetxt('ZT.txt', ZT.todense())

        BiGraph = (Z + ZT.T) / 2
        U_old = U
        V_old = V
        BiGraph, U, V, evc, D1, D2 = svd2uv(BiGraph, c)

        fn1 = np.sum(evc.reshape(-1,  order = 'F')[:c])
        fn2 = np.sum(evc.reshape(-1,  order = 'F')[:(c+1)])
        # print('fn1: ' + str(fn1))
        # print('fn2: ' + str(fn2))

        if fn1 < c - zr:
            Ater = 0
            llambda = 2*llambda
        elif fn2 > c+1-zr:
            Ater = 0
            llambda = llambda / 2
            U = U_old
            V = V_old
        else:
            Ater += 1
            if Ater == 2:
                break
    
    print('csbg loop: ' + str(iter1))

    laMM = id1[:,0]
    # np.savetxt('BiGraph.txt', BiGraph)
    
    clusternum, laKMM = struG2la(BiGraph)
    print('cluster number: ' + str(clusternum))
    if clusternum != c:
        print('Can not find the correct cluster number: '+ str(c))

    isCov = (Ater == 2)
    obj = loss(distX,BiGraph,alpha,llambda,U,V)
    OBJ = np.hstack((OBJ, obj)) if OBJ.size else obj
    
    return laKMM, laMM, BiGraph,isCov, OBJ, alpha, llambda

if __name__ == "__main__":
    pass