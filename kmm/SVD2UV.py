'''
@Author: your name
@Date: 2019-11-10 00:41:48
@LastEditTime: 2019-11-25 11:37:41
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /KMM-Python/kmm/SVD2UV.py
'''
import numpy as np
import scipy as sp


def eig1(A, c=-1, isMax=-1, B=None):
    # A=args[0]
    # c=0
    if c == -1:
        c = A.shape[0]
        isMax=1
    elif c>A.shape[0]:
        c=A.shape[0]

    if isMax == -1:
        isMax = 1

    if B == None:
        B = np.eye(A.shape[0])
    # if len(args)<2:
    #     c=A.shape[0]
    #     isMax=1
    # elif c>A.shape[0]:
    #     c=A.shape[0]

    # if len(args)<3:
    #     isMax=1

    # if len(args)<4:
    #     B=np.eye(A.shape[0])

    A=(A+A.T)/2

    # v,d=sp.eig(A,B)
    d, v = sp.linalg.eig(A, B)
    # f = open('d.txt', 'a')
    # np.savetxt(f, d, fmt = '%.4f', delimiter=',')
    # f.close()
    # d = np.diag(d)

    # f = open('v.txt', 'a')
    # np.savetxt(f, v, fmt = '%.4f', delimiter=',')
    # f.close()
    
    # v, d = np.linalg.eig(A, B)
    # print('d: ', d.shape)
    # print('v: ', v.shape)
    # d=np.diag(d)
    d=abs(d)

    if isMax==0:
        d1=np.sort(d)
        idx=np.argsort(d)
    else:
        idx=np.argsort(-d)
        d1=d[idx]

    idx1=idx[:c]
    eigval=d[idx1]
    eigvec=v[:,idx1]

    eigval_full=d[idx]

    return eigvec,eigval,eigval_full



def svd2uv(Z,c):
    n,m=Z.shape
    Z=Z/Z.sum(axis=1).reshape(-1,1)
    # print('svd2uv: Z: ' + str(type(Z)))
    # f = open('ZZ.txt', 'a')
    # np.savetxt(f, Z, fmt = '%.4f', delimiter=',')
    # f.close()
    z1=Z.sum(axis=1)
    # f = open('z1.txt', 'a')
    # np.savetxt(f, z1, fmt = '%.4f', delimiter=',')
    # f.close()
    D1z=sp.sparse.spdiags(1/np.sqrt(z1).reshape(-1),0,n,n)  # may need reshape here
    # print('D1z: '+str(D1z.shape))
    
    z2=Z.sum(axis=0)
    # print('z2: ', str(z2.shape))
    D2z=sp.sparse.spdiags(1/np.sqrt(z2).reshape(-1),0,m,m)  #
    # np.savetxt('D2z.txt', D2z.todense())
    # print('D2z: '+str(D2z.shape))
    Z1=D1z*Z*D2z

    # f = open('Z1.txt', 'a')
    # np.savetxt(f, Z1, fmt = '%.4f', delimiter=',')
    # f.close()

    ########## don't know how to implement full ###################
    ########  LZ = full(Z1'*Z1);###################################
    ###############################################################
    LZ = Z1.T * Z1
    
    # f = open('LZ.txt', 'a')
    # np.savetxt(f, LZ, fmt = '%.4f', delimiter=',')
    # f.close()

    V,evc,_=eig1(LZ,c+1)
    evc = evc.reshape((-1, 1))
    # f = open('V.txt', 'a')
    # np.savetxt(f, V, fmt = '%.4f', delimiter=',')
    # f.close()
    # f = open('evc.txt', 'a')
    # np.savetxt(f, evc, fmt = '%.4f', delimiter=',')
    # f.close()
    # print('V: ', str(V.shape))
    # print('evc: ', str(evc.shape))
    V=V[:,:c]
    U=(Z1*V)/(np.ones((n,1))*np.sqrt(evc[:c]).T)
    # print('Z1: ' +  str(type(Z1)))
    # print('V: ' +  str(type(V)))
    U=np.sqrt(2)/2*U
    V=np.sqrt(2)/2*V

    # f = open('U.txt', 'a')
    # np.savetxt(f, U, delimiter=',')
    # f.close()
    # f = open('V.txt', 'a')
    # np.savetxt(f, V, delimiter=',')
    # f.close()
    return Z, U, V, evc, D1z, D2z



