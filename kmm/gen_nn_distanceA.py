import scipy.sparse as ss
import numpy as np
import gc
import pickle

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

