import os
import json
import numpy as np
from time import time
import argparse
import tensorflow as tf

# seed experiment
import random
SEED = 0
random.seed(SEED)
tf.set_random_seed(SEED)

from docrec.metrics.solution import neighbor_comparison
from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.compatibility.proposed import Proposed
from docrec.solver.solverconcorde import SolverConcorde

# ignore all future warnings (due to scikit-image)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def reconstruct():

    # parameters processing
    parser = argparse.ArgumentParser(description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, or cdip].'
    )
    parser.add_argument(
        '-a', '--arch', action='store', dest='arch', required=False, type=str,
        default='sn', help='Network architecture [sn or mn].'
    )
    parser.add_argument(
        '-t', '--thresh', action='store', dest='thresh', required=False, type=str,
        default='sauvola', help='Thresholding method [otsu or sauvola].'
    )
    parser.add_argument(
        '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
        default=None, help='Model identifier (tag).'
    )
    parser.add_argument(
        '-i', '--input-size', action='store', dest='input_size', required=False, nargs=2, type=int,
        default=[3000, 32], help='Network input size (H x W).'
    )
    parser.add_argument(
        '-v', '--vshift', action='store', dest='vshift', required=False, type=int,
        default=10, help='Vertical shift range.'
    )
    parser.add_argument(
        '-r', '--results-id', action='store', dest='results_id', required=False, type=str,
        default=None, help='Identifier of the results file.'
    )
    parser.add_argument(
        '-md', '--max-ndocs', action='store', dest='max_ndocs', required=False, type=int,
        default=5, help='Maximum number of documents to be mixed.'
    )
    parser.add_argument(
        '-nf', '--num-features', action='store', dest='num_features', required=False, type=int,
        default=64, help='Number of features.'
    )

    args = parser.parse_args()

    input_size = tuple(args.input_size)

    assert args.dataset in ['D1', 'D2', 'cdip']
    assert args.arch in ['sn', 'sn-bypass']
    assert args.thresh in ['otsu', 'sauvola']
    assert args.results_id is not None
    assert args.vshift in [0, 5, 10, 15, 20]
    assert input_size in [(3000, 32), (3000, 48), (3000, 64)]

    # system setup
    training_set = 'isri-ocr' if args.dataset == 'cdip' else 'cdip'
    model_id = args.model_id if args.model_id is not None else '{}-{}'.format(training_set, args.arch)
    weights_path = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model']
    algorithm = Proposed(
        args.arch, weights_path, args.vshift, args.input_size, num_classes=2,
        verbose=False, thresh_method=args.thresh, seed=SEED
    )
    solver = SolverConcorde(maximize=True, max_precision=2)

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

    # results / initial configuration
    save_dir = 'results/exp2_ablation'
    os.makedirs(save_dir, exist_ok=True)
    results = {
        'model_id': model_id,
        'k_init': args.max_ndocs, # number of docs
        'it': 1,                  # iteration
        'state': None,            # random number generator state
        'data': {}                # experimental results data
    }

    # recover experiment
    results_fname = '{}/{}.json'.format(save_dir, args.results_id)
    if os.path.exists(results_fname):
       results = json.load(open(results_fname))
    it = results['it']
    k_init = results['k_init']
    state = results['state']
    if state is not None:
        state = (state[0], tuple(state[1]), state[2])
        random.setstate(state)

    # strips information
    map_doc_strips = {doc: Strips(path=doc, filter_blanks=True) for doc in docs}

    # mapping strips to ids for each document
    num_strips = 0
    map_strips_ids = {}
    for doc in docs:
        size = len(map_doc_strips[doc].strips)
        map_strips_ids[doc] = list(range(num_strips, num_strips + size))
        num_strips += size

    compatibilities_glob = np.ma.array(np.empty((num_strips, num_strips), dtype=np.float32), mask=True)
    displacements_glob = np.ma.array(np.empty((num_strips, num_strips), dtype=np.int32), mask=True)

    total = int(((ndocs - args.max_ndocs + 1) + ndocs) * args.max_ndocs / 2) # sum of an arithmetic progression
    for k in range(k_init, 0, -1):
        results_k = []
        for offset in range(ndocs - k + 1):

            # select k consecutive documents
            picked_docs = docs[offset : offset + k]
            strips_list = [map_doc_strips[doc] for doc in picked_docs]
            strips = MixedStrips(strips_list, shuffle=False)

            # strips ids
            strips_ids = [id_ for doc in picked_docs for id_ in map_strips_ids[doc]]
            sizes = [len(map_strips_ids[doc]) for doc in picked_docs]

            # pairs to be ignored
            ignore_pairs = []
            for i, id_i in enumerate(strips_ids):
                for j, id_j in enumerate(strips_ids):
                    if i == j:
                        continue
                    if not compatibilities_glob.mask[id_i, id_j]:
                        ignore_pairs.append((i, j))

            # print(ignore_pairs, len(ignore_pairs))
            # compute compatibilities
            t0 = time()
            algorithm.run(strips, 0, ignore_pairs)
            compatibilities = algorithm.compatibilities.copy()
            displacements = algorithm.displacements.copy()
            comp_time = time() - t0

            # update global compatibilities
            for i, id_i in enumerate(strips_ids):
                for j, id_j in enumerate(strips_ids):
                    if i == j:
                        continue
                    # not in cache?
                    if compatibilities_glob.mask[id_i, id_j]:
                        compatibilities_glob[id_i, id_j] = compatibilities[i, j]
                        compatibilities_glob.mask[id_i, id_j] = False
                        displacements_glob[id_i, id_j] = displacements[i, j]
                        displacements_glob.mask[id_i, id_j] = False
                    # already in cache
                    else:
                        compatibilities[i, j] = compatibilities_glob[id_i, id_j]
                        displacements[i, j] = displacements_glob[id_i, id_j]

            # shuffling strips
            init_perm = list(range(len(strips_ids)))
            random.shuffle(init_perm)
            compatibilities = compatibilities[init_perm][:, init_perm]
            # displacements = displacements[init_perm][:, init_perm]

            # computing solution
            t0 = time()
            solver.solve(compatibilities)
            solution = solver.solution
            opt_time = time() - t0
            accuracy = neighbor_comparison(solution, init_perm, sizes)
            print('[{:.2f}%] k={}, offset={}, ndocs={} accuracy={:.2f}% inf_time={:.2f}s comp_time={:.2f}s opt_time={:.2f}s'.format(
                100 * (it / total), k, offset, ndocs, 100 * accuracy, algorithm.inference_time, comp_time, opt_time
            ))
            results_k.append({
                'docs': picked_docs,
                'solution': solution,
                'accuracy': accuracy,
                'init_perm': init_perm,
                'sizes': sizes,
                'comp_time': comp_time,
                'opt_time': opt_time,
                'displacements': displacements.tolist(),
                'inf_time': algorithm.inference_time
            })
            it += 1
        # dump results and current state
        results['k_init'] = k - 1
        results['it'] = it
        results['state'] = random.getstate()
        results['data'][k] = results_k
        json.dump(results, open(results_fname, 'w'))


if __name__ == '__main__':
    t0 = time()
    reconstruct()
    t1 = time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format((t1 - t0) / 60.0, t1 - t0))
