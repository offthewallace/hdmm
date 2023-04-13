import benchmarks
from hdmm import workload, templates
import argparse
import numpy as np
import benchmarks
import pickle
import os
import time


def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['approx'] = 'True'
    params['dataset'] = 'adult'
    params['workload'] = 1
    params['output'] = 'hderror_2023_04_12.csv'

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['census','cps','adult','loans'],help='dataset to use')
    parser.add_argument('--workload', choices=[1,2], type=int, help='workload to use')
    parser.add_argument('--approx', choices=['False','True'], help='use approximate DP')
    parser.add_argument('--output', help='path to save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    approx = args.approx == 'True'

    W = benchmarks.get_workload(args.dataset, args.workload)
    ns = get_domain(W)

    #temp1 = templates.DefaultKron(ns, approx)
    #temp2 = templates.DefaultUnionKron(ns, len(W.matrices), approx)
    temp3 = templates.Marginals(ns, approx)
    t0 = time.time()

    #loss1 = temp1.optimize(W)
    #loss2 = temp2.optimize(W)
    loss3 = temp3.optimize(W)
    t1 = time.time()

    losses = {}
    #losses['kron'] = np.sqrt(loss1 / W.shape[0])
    #losses['union'] = np.sqrt(loss2 / W.shape[0])
    losses['marg'] = np.sqrt(loss3 / W.shape[0])
    line = ' %.6f, %.6f' % (t1-t0,losses['marg'])
    print(line)

    if args.output is not None:
        with open(args.output, 'a') as f:
            for param in losses.keys():
                key = (args.dataset, args.workload, approx, param, losses[param],t1-t0)
                f.write('%s, %d, %s, %s, %.4f,%.6f\n' % key)
