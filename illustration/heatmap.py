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
import math
import argparse
import tensorflow as tf
import json
from skimage.filters import threshold_sauvola, threshold_otsu

from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.neural.models.squeezenet import SqueezeNet
from docrec.ndarray.utils import first_nonzero, last_nonzero


def extract_features(strip, input_size, thresh_method='sauvola'):
    ''' Extract image around the border. '''

    input_size_h, input_size_w = input_size
    image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
    thresh_func = threshold_sauvola if thresh_method == 'sauvola' else threshold_otsu
    thresh = thresh_func(image)
    thresholded = (image > thresh).astype(np.float32)
    
    image_bin = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last

    wl = input_size_w // 2
    wr = wl
    h, w, _ = strip.image.shape
    offset = int((h - input_size_h) / 2)

    # left image
    left_border = strip.offsets_l + 2
    left = np.ones((input_size_h, wl, 3), dtype=np.float32)
    for y, x in enumerate(left_border[offset : offset + input_size_h]):
        w_new = min(wl, w - x)
        left[y, : w_new] = image_bin[y + offset, x : x + w_new]

    # right image
    right_border = strip.offsets_r - 2
    right = np.ones((input_size_h, wr, 3), dtype=np.float32)
    for y, x in enumerate(right_border[offset : offset + input_size_h]):
        w_new = min(wr, x + 1)
        right[y, : w_new] = image_bin[y + offset, x - w_new + 1: x + 1]

    return left, right


def create_overlay(strip_left, strip_right, conv10, input_size, view_mode):
    # def create_overlay(strip_left, strip_right, conv10, input_size, disp, view_mode):

    # data
    # input_size = tuple(args.input_size)
    input_size_h, input_size_w = input_size
    wr = input_size_w // 2
    neg = conv10[:, :, 0]
    pos = conv10[:, :, 1]
    # neu = conv10[:, :, 2]
   
    maps = np.stack([0 * pos, pos, neg]) # BGR
    maps = maps / maps.max()
    # if view_mode == 'max':
    #     mask = maps == maps.max(axis=0, keepdims=True)
    #     maps[mask] = 1.0
    #     maps[~mask] = 0
    
    maps = (255 * np.transpose(maps, axes=(1, 2, 0))).astype(np.uint8)
    maps = cv2.resize(maps, dsize=(wr, input_size_h), interpolation=cv2.INTER_CUBIC)

    # left strip
    overlay_left = strip_left.image.copy()
    offset = (strip_left.h - input_size_h) // 2
    left_border, right_border = strip_left.offsets_l[offset : offset + input_size_h], strip_left.offsets_r[offset : offset + input_size_h]
    for y, (x1, x2) in enumerate(zip(left_border, right_border)):
        wr_ = min(x2 - x1 + 1, wr)
        overlay_left[y + offset, x2 - wr_ + 1 : x2 + 1] = maps[y, : wr_]

    # right strip
    overlay_right = strip_right.image.copy()
    offset = (strip_right.h - input_size_h) // 2
    left_border, right_border = strip_right.offsets_l[offset : offset + input_size_h], strip_right.offsets_r[offset : offset + input_size_h]
    for y, (x1, x2) in enumerate(zip(left_border, right_border)):
        wr_ = min(x2 - x1 + 1, wr)
        overlay_right[y + offset, x1 : x1 + wr_] = maps[y, : wr_]

    return overlay_left, overlay_right


def binary(strip, thresh_method='sauvola'):

    strip = strip.copy()
    image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
    thresh_func = threshold_sauvola if thresh_method == 'sauvola' else threshold_otsu
    thresh = thresh_func(image)
    thresholded = (255 * (image > thresh)).astype(np.uint8)
    thresholded = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last
    strip.image = cv2.bitwise_and(
        thresholded, cv2.cvtColor(strip.mask, cv2.COLOR_GRAY2RGB)
    )
    return strip


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Score.')
    parser.add_argument(
        '-d', '--doc', action='store', dest='doc', required=False, type=str,
        default='datasets/D1/artificial/D001', help='Document.'
    )
    parser.add_argument(
        '-al', '--alpha', action='store', dest='alpha', required=False, type=float,
        default=0.5, help='Alpha channel.'
    )
    parser.add_argument(
        '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
        default=None, help='Model identifier (tag).'
    )
    parser.add_argument(
        '-v', '--view-mode', action='store', dest='view_mode', required=False, type=str,
        default='normal', help='View mode.'
    )
    parser.add_argument(
        '-is', '--input-size', action='store', dest='input_size', required=False, nargs=2, type=int,
        default=[3000, 32], help='Network input size (H x W).'
    )
    parser.add_argument(
        '-vs', '--vshift', action='store', dest='vshift', required=False, type=int,
        default=10, help='Vertical shift.'
    )
    args = parser.parse_args()

    input_size = tuple(args.input_size)
    assert input_size in [(3000, 32), (3000, 48), (3000, 64)]
    assert args.view_mode in ['normal', 'max']
    
    # data
    input_size = tuple(args.input_size)
    input_size_h, input_size_w = input_size
    wr = input_size_w // 2
    images_ph = tf.placeholder(
        tf.float32, name='images_ph', shape=(None, input_size_h, input_size_w, 3) # channels last
    )
    batch = np.ones((2 * args.vshift + 1, input_size_h, input_size_w, 3), dtype=np.float32)

    # model
    model = SqueezeNet(images_ph, num_classes=2, mode='test', channels_first=False)
    sess = model.sess
    logits_op = model.output#[:, : -1] # exclude neutral
    conv10_op = model.view # (batch, feat_height, 1, 2) 

    # if args.norm_method == 'softmax':
    probs_op = tf.nn.softmax(logits_op)
    # else:
    #     probs_op = logits_op / tf.reduce_sum(logits_op, axis=1, keepdims=True)
    comp_op = tf.reduce_max(probs_op[:, 1])
    disp_op = tf.argmax(probs_op[:, 1]) - args.vshift

    sess.run(tf.global_variables_initializer())
    # weights_path = '{}/{}'.format(os.getcwd(), open('bestmodel_{}.txt'.format(args.model_id)).read().strip())
    weights_path = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model']
    model.load_weights(weights_path)

    base_path = 'illustration/heatmap/{}'.format(args.model_id)
    base_name = '-'.join(args.doc.split('/')[1 :])
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # solution / displacements
    dataset = args.doc.split('/')[1]
    if dataset == 'D3':
        dataset = 'cdip'
    params = '_'.join(args.model_id.split('_')[1 :])
    results_fname = 'ablation/results/{}_{}_{}.json'.format(dataset, params, args.vshift)
    results = json.load(open(results_fname))['data']['1']
    solution = None
    displacements = None
    init_perm = None
    for run in results:
        doc = run['docs'][0].split('/')[-1]
        if doc == os.path.splitext(args.doc.split('/')[-1])[0]:
            solution = run['solution']
            displacements = run['displacements']
            init_perm = run['init_perm']

    # displacements = [displacements[solution[i]][solution[i + 1]] for i in range(len(solution) - 1)]
    solution = [init_perm[s] for s in solution]
    displacements = [displacements[solution[i]][solution[i + 1]] for i in range(len(solution) - 1)]
    normal = [strip for strip in Strips(path=args.doc, filter_blanks=True).strips]
    shuffled = copy.deepcopy(normal)
    random.shuffle(shuffled)
    reconstructed = [copy.deepcopy(normal[s]) for s in solution]
    
    print(displacements)

    for text, strips in zip(['normal', 'shuffled', 'reconstructed'], [normal, shuffled, reconstructed]):
        print(args.doc, text)
        # strips = MixedStrips([strips_], shuffle=shuffle)
        # strips_bin = Strips(strips_list=[binary(strip) for strip in strips.strips])

        displacements = []

        # features
        features = []
        for strip in strips:
            left, right = extract_features(strip, (input_size_h, input_size_w), 'sauvola')
            features.append((left, right))

        # compute map for the strip sequence
        N = len(strips)
        displacements = []
        # print('comp= ', end='')
        for i in range(N - 1):
            # print(batch[:, :, : wr].shape)
            batch[:, :, : wr] = features[i][1]
            batch[args.vshift, :, wr :, :] = features[i + 1][0] # radius zero
            for r in range(1, args.vshift + 1):
                batch[args.vshift - r, : -r, wr :] = features[i + 1][0][r :]  # slide up
                batch[args.vshift + r, r : , wr :] = features[i + 1][0][: -r] # slide down

            probs, conv10, comp, disp = sess.run([probs_op, conv10_op, comp_op, disp_op], feed_dict={images_ph: batch})
            batch[:] = 1.0
            # probs, conv10, comp, disp, mask, mask_95 = sess.run([probs_op, conv10_op, comp_op, disp_op, mask_op, mask_op_95], feed_dict={images_ph: batch})
            displacements.append(disp)
            # print(comp, end=' ')

            # normal
            overlay_left, overlay_right = create_overlay(
                strips[i], strips[i + 1],
                conv10[disp + args.vshift],
                input_size,
                args.view_mode
            )
            strips[i].image = cv2.addWeighted(overlay_left, args.alpha, strips[i].image, 1 - args.alpha, 0)
            strips[i + 1].image = cv2.addWeighted(overlay_right, args.alpha, strips[i + 1].image, 1 - args.alpha, 0)
        print(displacements)  
        # normal
        strip = strips[0].copy()
        for i in range(1, N):
            strip.stack(strips[i], displacements[i - 1], filled=False)
        # print(shuffle, doc.split('/')[3])
        cv2.imwrite(
            '{}/{}-{}.jpg'.format(
                base_path,
                base_name,
                text
                # ,
                # args.view_mode
            ), strip.image
        )
    sess.close()