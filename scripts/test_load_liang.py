from parse import search
from multiprocessing import Pool, cpu_count
import numpy as np
import argparse
import os
import gc
import cv2
import sys
import json
import subprocess
import tempfile
from time import time

from docrec.metrics.solution import neighbor_comparison
from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips

import random
random.seed(0)  # <= reproducibility

# path to the document reassembly software
DOCREASSEMBLY_PATH_DIR = '/mnt/data/DocReassembly'
# template of the command software to be executed
CMD_TEMPLATE = 'bin/release/solver -t {} -n {} -c 1 -m 2 -s 1000 -r --word_conf_thres 60 --lambda0 0.5 --lambda1 0.7 --u_a 1 --filter_rate 0.7 --candidate_factor 4' 
# height of strip so that they have an uniform size (required by the DocReassembly software)
MAX_STRIP_HEIGHT = 3000
# maximum instance size (# of documents) to perform the test
MAX_NUM_DOCS = 5

# global mapping variable: doc filename => Strip object
map_doc_strips = {}

def solve(instance):
   
    result = instance.copy()
    strips_list = [map_doc_strips[doc] for doc in instance['docs']]
    strips = MixedStrips(strips_list, shuffle=True)
    prefix_dir = '{}/data/stripes/'.format(DOCREASSEMBLY_PATH_DIR)
    suffix_dir = '_{}'.format(len(strips.strips)) # to match the rule of the DocReassmeble software
    
    # create a temporary directory to hold the reconstruction instance data
    with tempfile.TemporaryDirectory(prefix=prefix_dir, suffix=suffix_dir) as tmpdirname:
        # record the current directory and move to the DocReassembly root directory
        curr_dir = os.path.abspath('.')
        os.chdir(DOCREASSEMBLY_PATH_DIR)

        # case (instance) name is the basename of the directory without the final _<n>, where n is the
        # number of strips. DocReassmebly will concatenate the path data/stripes/<case_name>
        # with the parameter <n>.
        case_name = os.path.basename(tmpdirname).replace(suffix_dir, '')
        
        # set the command to be executed (replace open parameters in the template string)
        cmd = CMD_TEMPLATE.format(case_name, len(strips.strips))
        cmd = cmd.split() # split command to put in the format of the subprocess system call format

        # copy strips' images into the temporary directory
        for i, strip in enumerate(strips.strips):
            cv2.imwrite('{}/{}.png'.format(tmpdirname, i), strip.image[: MAX_STRIP_HEIGHT, :, :: -1])
        
        # write the order file (ground-truth)
        order = len(strips.strips) * ['0'] # inverted init perm (which piece should be in each position?)
        for pos, element in enumerate(strips.init_perm):
            order[element] = str(pos)
        open('{}/order.txt'.format(tmpdirname), 'w').write('\n'.join(order))
        # while(1): pass
        
        # run the software
        with open(os.devnull, 'w') as devnull:
            output = str(subprocess.check_output(cmd))#, stderr=devnull))
        os.chdir(curr_dir) # return to the original directory
    
    sizes = instance['sizes']
    solution = [int(s) for s in search('Composed order: {} \\n', output).fixed[0].split()]
    result['opt_time'] = float(search('Computation time: {}s', output).fixed[0])
    result['accuracy'] = neighbor_comparison(solution, strips.init_perm, sizes)
    return result


def test():

    # reconstruction documents of the dataset D2
    docs = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
    
    # pick MAX_NUM_DOCS documents
    random.shuffle(docs)
    docs = docs[: MAX_NUM_DOCS]
    
    # load strips
    strips_list = [Strips(path=doc, filter_blanks=True) for doc in docs]
    
    # each individual strip of a document is assigned an unique ID.
    # map document filename into the ids of such document strips
    map_doc_ids = {}
    cum = 0
    for doc, strips in zip(docs, strips_list):
        size = len(strips.strips)
        map_doc_strips[doc] = strips  # map_doc_strips is a global variable
        map_doc_ids[doc] = list(range(cum, cum + size))
        cum += size

    # main loop
    start = time()
    for k in range(1, MAX_NUM_DOCS + 1):
        # picking k documents
        picked_docs = docs[ : k]

        # union of ids of the picked docs' strips
        sizes_k = []
        for doc in picked_docs:
            sizes_k.append(len(map_doc_ids[doc]))

        # instances
        instance = {
            'docs': picked_docs,
            'accuracy': None,
            'sizes': sizes_k,
            'opt_time': None
        }
        result = solve(instance)
        print('k={} accuracy={:.2f}% opt_time={:.2f}s'.format(
            k, 100 * result['accuracy'], result['opt_time']
        ))


if __name__ == '__main__':
    t0 = time()
    test()
    t1 = time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format(
        (t1 - t0) / 60.0, t1 - t0))
