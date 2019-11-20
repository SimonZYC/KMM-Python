import numpy as np
import scipy as sp


def eig1(*args):
    A=args[0]
    c=0

    if len(args)<2:
        c=A.shape[0]
        isMax=1
    elif c>A.shape[0]:
        c=A.shape[0]

    if len(args)<3:
        isMax=1

    if len(args)<4:
        B=np.eye(A.shape[0])

    A=(A+A.T)/2

    v,d=sp.eig(A,B)
    d=np.diag(d)
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

    return eigvec,egival,eigval_full



def svd2uv(Z,c):
    n,m=Z.shape
    Z=Z/Z.sum(axis=1).reshape(-1,1)

    z1=Z.sum(axis=1)
    D1z=sp.sparse.spdiags(1/np.sqrt(z1).reshape(-1,1),0,n,n)  # may need reshape here

    z2=Z.sum(axis=0)
    D2z=sp.sparse.spdiags(1/np.sqrt().reshape(1,-1),0,m,m)  #

    Z1=D1z*Z*D2z

    ########## don't know how to implement full ###################
    ########  LZ = full(Z1'*Z1);###################################
    ###############################################################
    LZ = Z1.todense()

    V,evc,_=eig1(LZ,c+1)
    V=V[:,:c]
    U=(Z1*V)/(np.ones((3,1))*np.sqrt(evc[:c]).T)

    U=np.sqrt(2)/2*U
    V=np.sqrt(2)/2*V

    return Z, U, V, evc, D1z, D2z



