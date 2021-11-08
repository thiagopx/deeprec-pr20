# python -m illustration.heatmap --model-id isri-ocr-sn-rneu=0.05 --shuffle True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import random
import numpy as np
import os
import cv2
import json
import pyvips
import matplotlib.pyplot as plt
from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips

if __name__ == '__main__':

    run = json.load(open('comparison/proposed/cdip.json', 'r'))['data']['100'][0]
    matrix = json.load(open('comparison/proposed/cdip_matrix.json', 'r'))
    displacements = matrix['displacements']
    sizes = matrix['sizes']
    cdip_docs = ['datasets/D3/mechanical/D{:03}'.format(i) for i in range(1, 101)] # cdip
    picked_docs = run['docs']
    print(sizes)

    cdip_strips = [Strips(path=doc) for doc in cdip_docs]
    for doc, strips in zip(cdip_docs, cdip_strips):
        print(doc, len(strips))
    
    # map_strips_local_global = {}
    # i = 0
    # cum_sizes = np.cumsum(sizes)
    # for doc in picked_docs:
    #     pos = cdip_docs.index(doc)
    #     for j in range(cum_sizes[pos] - sizes[pos], cum_sizes[pos]):
    #         map_strips_local_global[i] = j
    #         i += 1


    # init_perm = run['init_perm']
    # solution = [init_perm[s] for s in run['solution']]
    # displacements = [displacements[map_strips_local_global[prev]][map_strips_local_global[curr]] for prev, curr in zip(solution[: -1], solution[1 :])]
    # strips = MixedStrips([Strips(path=doc) for doc in picked_docs], shuffle=False)
    # reconstructed = strips.image(solution, displacements, False)
    # img_pyvisp = pyvips.Image.new_from_array(reconstructed)
    # img_pyvisp.dzsave('illustration/reconstruction_s-cdip', suffix='.png')
    # # # plt.imsave('illustration/reconstruction_example.pdf', reconstructed[:: 4, :: 4])
    # # cv2.imwrite('illustration/reconstruction_example.jpg', reconstructed[:: 4, :: 4, :: -1])
    # # print(run['accuracy'])

