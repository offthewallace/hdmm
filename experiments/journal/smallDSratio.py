from hdmm import workload, templates
import time
import numpy as np
from IPython import embed
import gc
import argparse
import numpy as np
import benchmarks
import pickle
import os
import time
from scipy.sparse.linalg import LinearOperator, eigs,svds
from hdmm.matrix import Identity
from scipy.sparse import coo_matrix
import math 
from hdmm.error import per_query_error,per_query_error2,per_query_error_sampling2


def calculate_V(A, W):
    # using matrix.py's implementation but dont know if the __mul__ is correct
    # TODO: make sure it run into the right __mul__ function
    #A psedo inverse
    A1 = A.pinv()
    #WA's psedo inverse
    product = W @ A1
    #(WA')T
    inverse_matrix = product.T
    #(WA'(WA')T)
    product_result = product@inverse_matrix
    #time1 = time.time()
    #v_result=getmaxDiag(product_result)
    #time2=
    #v = product_result @ diag_vec
    v = product_result.diag()
    
    return v.max()

def calculate_V_v2(A, W):
    # written in pure numpy to verify the result
    #stuck in here 
    A1 = np.linalg.pinv(A.dense_matrix())
    W=W.dense_matrix()
    product = np.matmul(W, A1)
    transpose_matrix = product.T
    v = np.matmul(product, transpose_matrix)
    v = np.diag(v).max()
    return v

def getratio(fileName,dims=[0,1,2,3]):
    #with open('scalability_ls_0410.csv', 'w') as f:
    #with open(fileName, 'w') as f:
    #    f.write('n,Kronecker,Marginals,Marginaloss\n')

    for n in [2,4,8,16,32,64,128]:
       
        ns=tuple([n]*5)
        W = workload.DimKMarginals(ns, dims)
        #temp = templates.Marginals([n]*5)
        temp = templates.Marginals(ns, True)

        loss=temp.optimize(W)
        A = temp.strategy()
        A.weights = A.weights.astype(np.float32)
        A.dtype = np.float32
        #W.weights = W.weights.astype(np.float32)
        #W.dtype = np.float32
        #y = np.zeros(A.shape[0], dtype=np.float32)
        #t2 = time.time()
        #AtA1 = A.gram().pinv()
        #AtA1.weights = AtA1.weights.astype(np.float32)
        #AtA1.dtype = np.float32
        #At = A.T
        #At.dtype = np.float32
        #A1 = AtA1 @ At
        #A1.dot(y)
        #t3 = time.time()
        #v1=calculate_V_v2(A, W)
        #v2=per_query_error2(W,A)
        v=per_query_error_sampling2(W,A)
        #print(v)
        lossout = np.sqrt(loss / W.shape[0])
        with open(fileName,'a') as f:
            line = '%d, %.6f, %.6f' % (n, v.min(),lossout)
            print(line)
            f.write(line+'\n')

if __name__ == '__main__':
    getratio("ratio_0424_2.csv")