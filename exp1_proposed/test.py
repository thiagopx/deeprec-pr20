import os
import sys
import json
from time import time

import random
seed = 0
random.seed(seed) # <= reproducibility

import argparse
import numpy as np
from multiprocessing import Pool, RawArray

from docrec.metrics.solution import neighbor_comparison
from docrec.strips.strips import Strips

# system setup
from docrec.solver.solverconcorde import SolverConcorde
from docrec.solver.solvernn import SolverNN

solver = SolverConcorde(maximize=True, max_precision=3)

def solve(instance):

    result = instance.copy()
    compatibilities_k = instance['compatibilities']
    k = instance['k']
    offset = instance['offset']

    # optimization
    t0 = time()
    solver.solve(compatibilities_k, fname='/tmp/tsp{}_{}.tsp'.format(k, offset))
    opt_time = time() - t0

    result['compatibilities'] = None # save memory
    result['solution'] = solver.solution
    result['opt_time'] = opt_time
    result['accuracy'] = neighbor_comparison(solver.solution, result['init_perm'], result['sizes'])
    return result


def test():

    global solver
    global extra_args

    # parameters processing
    parser = argparse.ArgumentParser(description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, or cdip].'
    )
    parser.add_argument(
        '-np', '--nproc', action='store', dest='nproc', required=False, type=int,
        default=10, help='Number of processes.'
    )
    args = parser.parse_args()

    assert args.dataset in ['D1', 'D2', 'cdip']

    # reconstruction instances
    if args.dataset == 'D1':
        docs = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
    elif args.dataset == 'D2':
        docs = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
    else:
        docs = ['datasets/D3/mechanical/D{:03}'.format(i) for i in range(1, 101)] # cdip
    # shuffle documents
    random.shuffle(docs) # this should be considered in other scripts
    ndocs = len(docs)

    # global compatibility matrix
    compatibilities = json.load(open('results/exp1_proposed/{}_matrix.json'.format(args.dataset), 'r'))
    compatibilities = np.array(result_glob['compatibilities']) # generated with permutated documents (same seed)

    # strips ids for each document
    cum = 0
    ids_strips = {}
    for doc in docs:
        size = len(Strips(path=doc, filter_blanks=True).strips)
        ids_strips[doc] = list(range(cum, cum + size))
        cum += size

    # results / initial configuration
    ndocs_per_iter = [1, 2, 3, 4, 5] + list(range(10, ndocs + 1, 5))
    ndocs_per_iter.reverse() # process in reverve order
    results = {
        'matrix_id': '{}_matrix'.format(args.dataset),
        # 'k_init': 0,                                # index of ndocs_per_iter where the process should start
        'backup_iter': 0,                             # last iteration
        'state': None,                                # random number generator state
        'data': {str(k) : [] for k in ndocs_per_iter} # experimental results data
    }

    results_fname = 'results/exp1_proposed/{}.json'.format(dir_name, args.dataset)
    if os.path.exists(results_fname):
       results = json.load(open(results_fname))
    it = results['backup_iter']

    state = results['state']
    if state is not None:
        state = (state[0], tuple(state[1]), state[2])
        random.setstate(state)

    # main loop
    with Pool(processes=args.nproc) as pool:
        start = time()
        total = sum([ndocs - k + 1 for k in ndocs_per_iter])
        combs = [(k, offset) for k in ndocs_per_iter for offset in range(ndocs - k + 1)]
        instances = []
        for k, offset in combs[it :]:
            # picking documents
            picked_docs = docs[offset : offset + k]

            # union of ids
            ids_strips_k = []
            sizes_k = []
            for doc in sorted(picked_docs):
                ids_strips_k += ids_strips[doc]
                sizes_k.append(len(ids_strips[doc]))

            # crop compatibilities
            compatibilities_k = compatibilities[ids_strips_k][:, ids_strips_k]

            # shuffle strips
            N = compatibilities_k.shape[0]
            init_perm_k = list(range(N))
            random.shuffle(init_perm_k)

            # shuffle compatibilites
            compatibilities_k = compatibilities_k[init_perm_k][:, init_perm_k]

            # update instances
            instance = {
                'k': str(k),
                'offset': offset,
                'docs': picked_docs,
                'solution': None,
                'accuracy': None,
                'init_perm': init_perm_k,
                'sizes': sizes_k,
                'compatibilities': compatibilities_k,
                'opt_time': None
            }
            instances.append(instance)

            # run when the buffer size is greater then args.nproc or when the last iteration is achieved
            if (len(instances) == args.nproc) or (it + len(instances) == total):
                print('Iterations {}-{}/{}'.format(it + 1, it + len(instances), total), end=' ')
                results_buffer = pool.map(solve, instances)
                it += len(instances)
                instances = [] # reset instances
                elapsed = time() - start
                predicted = elapsed * (total - it) / it
                print(':: elapsed={:.2f}s :: predicted={:.2f}s'.format(elapsed, predicted))
                for result in results_buffer:
                    results['data'][result['k']].append(result)
                    print('    => k={} offset={} accuracy={:.2f}% opt_time={:.2f}s'.format(
                        result['k'], result['offset'], 100 * result['accuracy'], result['opt_time']
                    ))

                # dump results and current state
                results['backup_iter'] = it
                results['state'] = random.getstate()
                json.dump(results, open(results_fname, 'w'))


if __name__ == '__main__':
    t0 = time()
    test()
    t1 = time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format((t1 - t0) / 60.0, t1 - t0))