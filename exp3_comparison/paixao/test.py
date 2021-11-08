''' Usage: python -m test.paixao-incremental --dset <D1 or D2> --nruns <integer>'''
import os
import gc
import sys
import json
from time import time

import random
random.seed(0) # <= reproducibility

import argparse
import numpy as np
from multiprocessing import Pool, cpu_count, RawArray
from multiprocessing.pool import ThreadPool

from docrec.metrics.solution import neighbor_comparison
from docrec.strips.stripschar import StripsChar
from docrec.strips.mixedstripschar import MixedStripsChar
from docrec.compatibility.paixao import Paixao
from docrec.solver.solverconcorde import SolverConcorde

strips_dict = {}
MAX_PROC = 50
# proc_dict = {k: k / max_proc for k in range(1, 6)}
solver = SolverConcorde(maximize=True, max_precision=3)


def load_strips(doc):
    ''' Load strips object. '''

    return StripsChar(path=doc, filter_blanks=True)


def solve(instance):

    result = instance.copy()

    # compatibilities calculation
    strips_list = [strips_dict[doc] for doc in instance['docs']]
    strips = MixedStripsChar(strips_list, shuffle=True)
    seed = hash(doc.id) % 4294967295 # doc is last one in instance['docs']
    algorithm = Paixao(
        FFm=1.0, CC=1.0, EC=0.0, p=-0.2, gamma=0.0125,
        threshold=0.5, ns=100, seed=seed, max_cache_size=100000000, verbose=False
    )
    t0 = time()
    compatibilities_k = algorithm.run(strips).matrix
    comp_time = time() - t0
    gc.collect()
    k = instance['k']
    offset = instance['offset']

    # solving
    t0 = time()
    solutions = []
    for comp in compatibilities_k:
        solver.solve(comp, fname='/tmp/tsp_{}-{}.tsp'.format(k, offset))
        solutions.append(solver.solution)
    opt_time = time() - t0

    best = None
    accuracy = float('nan')
    max_accuracy = -1
    for solution in solutions:
        if solution is not None:
            accuracy = neighbor_comparison(solution, strips.init_perm, result['sizes'])
            if accuracy > max_accuracy:
                best = solution
                max_accuracy = accuracy

    #result['compatibilities'] = None # save memory
    result['init_perm'] = strips.init_perm
    result['solution'] = best
    result['comp_time'] = comp_time
    result['opt_time'] = opt_time
    result['accuracy'] = max_accuracy
    return result


def test():

    # parameters processing
    parser = argparse.ArgumentParser(description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, or cdip].'
    )
    # parser.add_argument(
    #     '-np', '--nproc', action='store', dest='nproc', required=False, type=int,
    #     default=10, help='Number of processes.'
    # )
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
    random.shuffle(docs)
    ndocs = len(docs)

    # load strips
    print('loading strips...', end=' ')
    sys.stdout.flush()
    t0 = time()
    nproc = min(MAX_PROC, cpu_count())
    print(nproc)
    with Pool(processes=nproc) as pool:
        strips_list = pool.map(load_strips, docs)
    load_time = time() - t0
    print('done! elapsed time={:.2f} seconds'.format(load_time))
    sys.stdout.flush()

    # strips ids for each document
    cum = 0
    ids_strips = {}
    for doc, strips in zip(docs, strips_list):
        size = len(strips.strips)
        strips_dict[doc] = strips # strips_dict is a global variable
        ids_strips[doc] = list(range(cum, cum + size))
        cum += size

    # initial configuration
    ndocs_per_iter = [5, 4, 3, 2, 1]
    results = {
        'load_time': load_time,                       # time elapsed to load the strips
        'backup_iter': 0,                             # last iteration
        'state': None,                                # random number generator state
        'data': {str(k) : [] for k in ndocs_per_iter} # experimental results data
    }

    # save directory
    dir_name = 'comparison/paixao'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    results_fname = '{}/{}.json'.format(dir_name, args.dataset)
    if os.path.exists(results_fname):
       results = json.load(open(results_fname))
    it = results['backup_iter']
    # k_init = results['k_init']
    state = results['state']
    if state is not None:
        state = (state[0], tuple(state[1]), state[2])
        random.setstate(state)

    # main loop
    start = time()
    total = sum([ndocs - k + 1 for k in ndocs_per_iter])
    combs = [(k, offset) for k in ndocs_per_iter for offset in range(ndocs - k + 1)]
    instances = []
    proc_load = 0
    for k, offset in combs[it :]:
        # picking documents
        picked_docs = sorted(docs[offset : offset + k])

        # union of ids
        # ids_strips_k = []
        sizes_k = []
        for doc in picked_docs:
            # ids_strips_k += ids_strips[doc]
            sizes_k.append(len(ids_strips[doc]))

        # update instances
        instance = {
            'k': str(k),
            'offset': offset,
            'docs': picked_docs,
            'solution': None,
            'accuracy': None,
            'init_perm': None,
            'sizes': sizes_k,
            # 'compatibilities': None,
            'comp_time': None,
            'opt_time': None
        }
        instances.append(instance)
        proc_load += k / MAX_PROC

        # run when the buffer size is greater then args.nproc or when the last iteration is achieved
        if (proc_load >= 0.99) or (it + len(instances) == total):
            print('Iterations {}-{}/{} proc. load={:.2f}%'.format(it + 1, it + len(instances), total, 100 * proc_load), end=' ')
            with Pool(processes=len(instances)) as pool:
                results_buffer = pool.map(solve, instances)
            it += len(instances)
            instances = [] # reset instances
            proc_load = 0
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
