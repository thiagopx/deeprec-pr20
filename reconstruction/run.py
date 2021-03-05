import os
import sys
import json
import cv2
from time import time

import random
SEED = 0
random.seed(SEED) # <= reproducibility
NUM_CLASSES = 2

import argparse
import numpy as np

from docrec.metrics.solution import neighbor_comparison
from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.compatibility.proposed import Proposed
from docrec.pipeline import Pipeline
from docrec.solver.solverconcorde import SolverConcorde
from docrec.solver.solvernn import SolverNN


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test():

    # parameters processing
    parser = argparse.ArgumentParser(description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--doc', action='store', dest='doc', required=False, type=str,
        default='datasets/D1/mechanical/D001', help='Document path.'
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
        '-s', '--solver', action='store', dest='solver', required=False, type=str,
        default='conc', help='Solver.'
    )
    parser.add_argument(
        '-o', '--offset', action='store', dest='offset', required=False, type=int, default=None,
        help='Vertical offset for feature extraction'
    )
    args = parser.parse_args()
    
    assert args.solver in ['conc', 'nn']
    
    offset = args.offset if args.offset != -1 else None
    
    # setting reconstruction pipeline
    weights_path = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model']
    algorithm = Proposed(
        args.arch, weights_path, args.vshift, args.input_size, num_classes=NUM_CLASSES,
        verbose=True, thresh_method=args.thresh, seed=SEED, offset=offset
    )
    solver = SolverNN(maximize=True, mode='repetitive', seed=SEED) if args.solver == 'nn' else SolverConcorde(maximize=True, max_precision=3)
    pipeline = Pipeline(algorithm, solver)

    print('Reconstructing document {}'.format(args.doc), end='')
    strips = Strips(path=args.doc, filter_blanks=True)
    strips_mx = MixedStrips([strips], shuffle=True) # The class MixedStrips enables shuffling the strips
    solution, compatibilities, displacements = pipeline.run(strips_mx)
    accuracy = neighbor_comparison(solution, strips_mx.init_perm, strips_mx.sizes)
    print('accuracy={:.2f}'.format(100 * accuracy))
    
    # reconstruction image
    displacements = [displacements[solution[i]][solution[i + 1]] for i in range(len(solution) - 1)]
    solution = [strips_mx.init_perm[s] for s in solution]
    reconstruction = strips.image(solution, displacements, True)
    cv2_image = reconstruction[..., :: -1] # RGB to BGR
    cv2.imwrite('reconstruction/reconstruction.jpg', cv2_image)


if __name__ == '__main__':
    t0 = time()
    test()
    t1 = time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format((t1 - t0) / 60.0, t1 - t0))
