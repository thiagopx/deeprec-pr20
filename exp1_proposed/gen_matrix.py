import os
import json
from time import time
import random
import argparse

# seed experiment
SEED = 0
random.seed(SEED)

from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.compatibility.proposed import Proposed


def main():

    # parameters processing
    parser = argparse.ArgumentParser(description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, or cdip].'
    )
    parser.add_argument(
        '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
        default=None, help='Model identifier (tag).'
    )
    args = parser.parse_args()

    assert args.dataset in ['D1', 'D2', 'cdip']

    # save directory
    save_dir = 'results/exp1_proposed'
    os.makedirs(save_dir, exist_ok=True)

    results_fname = '{}/{}_matrix.json'.format(save_dir, args.dataset)

    # system setup
    weights_path = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model']
    algorithm = Proposed(
        'sn', weights_path, 10, (3000, 32), num_classes=2,
        verbose=True, thresh_method='sauvola', seed=SEED
    )

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

    # build all
    strips_list = [Strips(path=doc, filter_blanks=True) for doc in docs]
    strips = MixedStrips(strips_list, shuffle=False)
    t0 = time()
    algorithm.run(strips, 0)
    comp_time = time() - t0
    print('ndocs={} inf_time={:.2f}% comp_time={:.2f}s'.format(
        ndocs, algorithm.inference_time, comp_time
    ))
    results = {
        'init_perm': strips.init_perm,
        'sizes': strips.sizes,
        'compatibilities': algorithm.compatibilities.tolist(),
        'displacements': algorithm.displacements.tolist(),
        'inf_time': algorithm.inference_time,
        'comp_time': comp_time
    }
    json.dump(results, open(results_fname, 'w'))


if __name__ == '__main__':

    t0 = time()
    main()
    t1 = time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format((t1 - t0) / 60.0, t1 - t0))
