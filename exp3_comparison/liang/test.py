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

# template of the command software to be executed
# t: case_name [str]
# n: number of strips (usually the instance is <t>_<n>) [int]
# c: composition mode [int]
#     0: GREEDY
#     1: GCOM
#     2: GREEDY_GCOM
#     3: GT
#     4: USER
# m: metric mode
#     0: PIXEL
#     1: CHAR
#     2: WORD
# s: number of samples for word path composition
# r: flag that indicates whether the instance is real-shredded
# u: flag that makes the software use the masks (pre-segmentation)
CMD_TEMPLATE = 'bin/release/solver -l {} -p {} -t {} -n {} -c 1 -m 2 -s 10000 {} --word_conf_thres 60 --lambda0 0.5 --lambda1 0.7 --u_a 1 --filter_rate 0.7 --candidate_factor 4'
# height of strip so that they have an uniform size (required by the DocReassembly software)
MAX_STRIP_HEIGHT = 3000

# global mapping variable: doc filename => Strip object
map_doc_strips = {}


def solve(instance, lang='eng', soft_path='/mnt/data/DocReassembly', use_mask=False):

    result = instance.copy()
    strips_list = [map_doc_strips[doc] for doc in instance['docs']]
    strips = MixedStrips(strips_list, shuffle=True)

    # path where the shreds will be placed:
    # tempdirma = prefix + case_name [randomly generated] + suffix
    prefix_dir = '{}/data/stripes/'.format(soft_path)
    suffix_dir = '_{}'.format(len(strips.strips)) # append the to match the rule of the DocReassmeble software

    # shreds' segmentation flag
    seg_flag = '-u' if use_mask else '-r' # -r handles with real instances based on hard-coded heuristic

    # create a temporary directory to hold the reconstruction instance data and run the experiment
    failed = False
    with tempfile.TemporaryDirectory(prefix=prefix_dir, suffix=suffix_dir) as tmpdirname:
        # record the current directory and move to the DocReassembly root directory
        curr_dir = os.path.abspath('.')
        os.chdir(soft_path)

        # case (instance) name is the basename of the directory without the final _<n>, where n is the
        # number of strips. DocReassmebly will concatenate the path data/stripes/<case_name>
        # with the parameter <n>.
        case_name = os.path.basename(tmpdirname).replace(suffix_dir, '')

        # set the command to be executed (replace open parameters in the template string)
        tess_model_path = '{}/data/tesseract_model'.format(soft_path)
        cmd = CMD_TEMPLATE.format(lang, tess_model_path, case_name, len(strips.strips), seg_flag)
        print(cmd)
        cmd = cmd.split()  # split command to put in the format of the subprocess system call format

        # copy strips' images into the temporary directory
        for i, strip in enumerate(strips.strips):
            cv2.imwrite('{}/{}.png'.format(tmpdirname, i),
                        strip.image[: MAX_STRIP_HEIGHT, :, :: -1])
            cv2.imwrite('{}/{}m.png'.format(tmpdirname, i),
                        strip.mask[: MAX_STRIP_HEIGHT])

        # write the order file (ground-truth)
        # inverted init perm (which piece should be in each position?)
        order = len(strips.strips) * ['0']
        for pos, element in enumerate(strips.init_perm):
            order[element] = str(pos)
        open('{}/order.txt'.format(tmpdirname), 'w').write('\n'.join(order))

        # run the software
        output = ''
        try:
            output = str(subprocess.check_output(cmd))
        except:
            failed = True

        os.chdir(curr_dir)

    # print(output)
    result['comp_time'] = 0.0
    result['init_perm'] = strips.init_perm
    if not failed:
        result['opt_time'] = float(search('Computation time: {}s', output).fixed[0])
        result['solution'] = [int(s) for s in search('Composed order: {} \\n', output).fixed[0].split()]
        result['accuracy'] = neighbor_comparison(result['solution'], strips.init_perm, result['sizes'])
    else:
        result['opt_time'] = 0.0
        result['solution'] = None
        result['accuracy'] = 0.0
    return result


def test():
    # parameters processing
    parser = argparse.ArgumentParser(
        description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, or cdip].'
    )
    parser.add_argument(
        '-s', '--soft-path', action='store', dest='soft_path', required=False, type=str,
        default='/mnt/data/DocReassembly', help='Path to the software which implements the Liang method.'
    )
    parser.add_argument(
        '-n', '--num-threads', action='store', dest='num_threads', required=False, type=int,
        default=24, help='Number of threads (OpenMP).'
    )
    parser.add_argument(
        '-m', '--use-mask', action='store', dest='use_mask', required=False, type=str,
        default='False', help='Use pre-segmented mask for the shreds.'
    )
    args = parser.parse_args()

    assert args.dataset in ['D1', 'D2', 'cdip']
    assert args.use_mask in ['True', 'False']

    # typecasting
    use_mask = True if args.use_mask == 'True' else False

    # create base dir where strips will be places
    os.makedirs('{}/data/stripes'.format(args.soft_path), exist_ok=True)

    # reconstruction instances
    lang = 'eng'
    if args.dataset == 'D1':
        lang = 'por'
        docs = [
            'datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
    elif args.dataset == 'D2':
        docs = [
            'datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
    else:
        # cdip
        docs = [
            'datasets/D3/mechanical/D{:03}'.format(i) for i in range(1, 101)]
    # shuffle documents
    random.shuffle(docs)
    ndocs = len(docs)

    # load strips
    print('loading strips...', end=' ')
    sys.stdout.flush()
    t0 = time()
    strips_list = [Strips(path=doc, filter_blanks=True) for doc in docs]
    load_time = time() - t0
    print('done! elapsed time={:.2f} seconds'.format(load_time))
    sys.stdout.flush()

    # each individual strip of a document is assigned an unique ID.
    # map document filename into the ids of such document strips
    map_doc_ids = {}
    cum = 0
    for doc, strips in zip(docs, strips_list):
        size = len(strips.strips)
        map_doc_strips[doc] = strips  # map_doc_strips is a global variable
        map_doc_ids[doc] = list(range(cum, cum + size))
        cum += size

    # initial configuration
    ndocs_per_iter = [1, 2, 3, 4, 5]
    results = {
        'load_time': load_time,  # time elapsed to load the strips
        'backup_iter': 0,        # last iteration
        'state': None,           # random number generator state
        # experimental results data
        'data': {str(k): [] for k in ndocs_per_iter}
    }

    # save directory
    dir_name = 'comparison/liang'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    results_fname = '{}/{}_threads={}_use-mask={}.json'.format(
        dir_name, args.dataset, args.num_threads, args.use_mask
    )
    if os.path.exists(results_fname):
        results = json.load(open(results_fname))
    it = results['backup_iter']
    state = results['state']
    if state is not None:
        state = (state[0], tuple(state[1]), state[2])
        random.setstate(state)

    # main loop
    start = time()
    total = sum([ndocs - k + 1 for k in ndocs_per_iter])
    combs = [(k, offset)
             for k in ndocs_per_iter for offset in range(ndocs - k + 1)]
    for k, offset in combs[it:]:
        # picking documents
        picked_docs = sorted(docs[offset: offset + k])

        # sizes (# of strips) of each picked document
        sizes_k = []
        for doc in picked_docs:
            sizes_k.append(len(map_doc_ids[doc]))

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

        # run the solver
        result = solve(instance, lang, args.soft_path, use_mask)
        results['data'][result['k']].append(result)
        print('    => k={} offset={} accuracy={:.2f}% opt_time={:.2f}s'.format(
            result['k'], result['offset'], 100 *
            result['accuracy'], result['opt_time']
        ), end='')
        if result['solution'] == None:
            print('[failed]', end='')

        it += 1
        elapsed = time() - start
        predicted = elapsed * (total - it) / it
        print(' :: elapsed={:.2f}s :: predicted={:.2f}s'.format(
            elapsed, predicted))

        # dump results and current state
        results['backup_iter'] = it
        results['state'] = random.getstate()
        json.dump(results, open(results_fname, 'w'))


if __name__ == '__main__':
    t0 = time()
    test()
    t1 = time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format(
        (t1 - t0) / 60.0, t1 - t0))
