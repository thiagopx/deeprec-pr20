import sys
import json
import numpy as np
from time import time
import argparse
import tensorflow as tf

import matplotlib.pyplot as plt

# seed experiment
import random
SEED = 0
random.seed(SEED)
tf.set_random_seed(SEED)

from docrec.metrics.solution import neighbor_comparison as accuracy
from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.compatibility.proposed import Proposed
from docrec.solver.solverconcorde import SolverConcorde
from docrec.pipeline import Pipeline


# parameters processing
parser = argparse.ArgumentParser(description='Demo: reconstructing 2 mixed documents.')
parser.add_argument(
    '-d1', '--doc1', action='store', dest='doc1', required=False, type=str,
    default='datasets/D3/mechanical/D009', help='Document 1 (of 2).'
)
parser.add_argument(
    '-d2', '--doc2', action='store', dest='doc2', required=False, type=str,
    default='datasets/D3/mechanical/D010', help='Document 2 (of 2).'
)
parser.add_argument(
    '-a', '--arch', action='store', dest='arch', required=False, type=str,
    default='sn', help='Network architecture [sn or mn].'
)
parser.add_argument(
    '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
    default='cdip_0.2_1000_32x32', help='Model identifier (directory in traindata).'
)
parser.add_argument(
    '-i', '--input-size', action='store', dest='input_size', required=False, nargs=2, type=int,
    default=[3000, 32], help='Overall networks input size (H x W) for test: (H x W/2) for each network'
)
parser.add_argument(
    '-v', '--vshift', action='store', dest='vshift', required=False, type=int,
    default=10, help='Vertical shift range.'
)
args = parser.parse_args()

# pipeline: compatibility algorithm + solver
weights_path = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model']
algorithm = Proposed(
    args.arch, weights_path, args.vshift, args.input_size, num_classes=2,
    verbose=False, thresh_method='sauvola', seed=SEED
)
solver = SolverConcorde(maximize=True, max_precision=3)
pipeline = Pipeline(algorithm, solver)

# load strips/shreds
print('1) Load strips')
strips1 = Strips(path=args.doc1, filter_blanks=True)
strips2 = Strips(path=args.doc2, filter_blanks=True)

# mixing strips
print('2) Mixing strips')
strips = MixedStrips([strips1, strips2], shuffle=True)
print('Shuffled order: ' + str(strips.init_perm))

print('3) Results')
solution, compatibilities, displacements = pipeline.run(strips)
displacements = [displacements[prev][curr] for prev, curr in zip(solution[: -1], solution[1 :])]
corrected = [strips.init_perm[idx] for idx in solution]
print('Solution: ' + str(solution))
print('Correct order: ' + str(corrected))
print('Accuracy={:.2f}%'.format(100 * accuracy(solution, strips.init_perm, strips.sizes)))
reconstruction = strips.image(order=solution, displacements=displacements)
plt.imshow(reconstruction)
plt.axis('off')
plt.show()