
# python -m illustration.heatmap --model-id isri-ocr-sn-rneu=0.05 --shuffle True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import json
import numpy as np
import os
import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
import math
import argparse
import random
import tensorflow as tf

from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.neural.models.squeezenet import SqueezeNet
from docrec.neural.models.reduced_squeezenet import RedSqueezeNet
from docrec.ndarray.utils import first_nonzero, last_nonzero


def create_overlay(image, view, input_size, view_mode, include_neutral=False):

    # data
    input_size_h, input_size_w = input_size
    wr = input_size_w // 2
    neg = view[:, :, 0]
    pos = view[:, :, 1]
    neu = view[:, :, 2]
    
    if include_neutral:
        maps = np.stack([neu, pos, neg]) # BGR
    else:
        maps = np.stack([0 * pos, pos, neg])

    if view_mode == 'global':
        maps = maps / maps.max()
    
    maps = (255 * np.transpose(maps, axes=(1, 2, 0))).astype(np.uint8)
    maps = cv2.resize(maps, dsize=(input_size_w, input_size_h), interpolation=cv2.INTER_CUBIC)

    maps_max = maps.copy()
    maps_max[maps != maps.max(axis=2, keepdims=True)] = 0
    return maps, maps_max


def binary(image, thresh_method='sauvola'):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_func = threshold_sauvola if thresh_method == 'sauvola' else threshold_otsu
    thresh = thresh_func(image)
    thresholded = (image > thresh)
    thresholded = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last
    return thresholded


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Score.')
    # parser.add_argument(
    #     '-d', '--doc', action='store', dest='doc', required=False, type=str,
    #     default='datasets/D1/artificial/D001', help='Document.'
    # )
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset.'
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
        default='norm', help='View mode.'
    )
    parser.add_argument(
        '-is', '--input-size', action='store', dest='input_size', required=False, nargs=2, type=int,
        default=[3000, 32], help='Network input size (H x W).'
    )
    parser.add_argument(
        '-s', '--shuffle', action='store', type=str, dest='shuffle',
        required=False, default='False', help='Shuffle image.'
    )
    parser.add_argument(
        '-a', '--arch', action='store', type=str, dest='arch',
        required=False, default='sn', help='Architecture.'
    )
    args = parser.parse_args()

    input_size = tuple(args.input_size)
    assert args.view_mode in ['norm', 'global']
    
    # data
    input_size_h, input_size_w = input_size
    wr = input_size_w // 2
    # images_ph = tf.placeholder(
    #     tf.float32, name='images_ph', shape=(None, input_size_h, input_size_w, 3) # channels last
    # )

    doc = cv2.imread('datasets/D1/full_images/D001.jpg', cv2.IMREAD_COLOR)
    height, width, channels = doc.shape
    images_ph = tf.placeholder(
        tf.float32, name='images_ph', shape=(None, height, width, channels) # channels last
    )

    acc = 0
    num_strips = 30
    dw = (width // num_strips) + 1
    strips = []
    while acc < width:
        strip = binary(doc[:, acc : acc + dw])
        acc += dw
        strips.append(strip)
    
    shuffled_strips = strips.copy()
    random.shuffle(shuffled_strips)
    
    image1 = np.hstack(strips)
    image2 = np.hstack(shuffled_strips)
    batch = np.array([image1, image2]).astype(np.float32)
    print(batch.shape)
            
    # model
    if args.arch == 'sn':
        model = SqueezeNet(images_ph, num_classes=3, mode='test', channels_first=False)
    else:
        model = RedSqueezeNet(images_ph, num_classes=3, mode='test', channels_first=False)

    sess = model.sess
    logits_op = model.output
    probs_op = tf.nn.softmax(logits_op, axis=3)
    comp_op = tf.reduce_sum(probs_op, [1, 2])
     # exclude neutral
    #logits_op = model.output[0]
    # view_op = model.view[0]
    view_op = probs_op
    # probs_op = model.probs
    #probs_op = tf.reduce_sum(model.probs[0], axis=[0, 1])
    
    base_path = 'illustration/bias'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    sess.run(tf.global_variables_initializer())
    model_id = 'cdip_test'
    weights_path = json.load(open('traindata/{}/info.json'.format(model_id), 'r'))['best_model']
    model.load_weights(weights_path)
    comp, view = sess.run([comp_op, view_op], feed_dict={images_ph: batch})
    for comp_, view_, image, label in zip(comp, view, batch, ['normal', 'shuffled']):
        image_ = (255 * image).astype(np.uint8)
        print(image_.shape)
        print(view_.shape)
        print('{} (-)={:5.2f}% (+)={:5.2f}% (N)={:5.2}%'.format(model_id, comp_[0], comp_[1], comp_[2]))
        overlay_mixed, overlay_max = create_overlay(image_, view_, (height, width), args.view_mode)
        image = cv2.addWeighted(overlay_mixed, args.alpha, image_, 1 - args.alpha, 0)
        cv2.imwrite('{}/{}-{}-mixed.jpg'.format(base_path, model_id, label), image)
        image = cv2.addWeighted(overlay_max, args.alpha, image_, 1 - args.alpha, 0)
        cv2.imwrite('{}/{}-{}-max.jpg'.format(base_path, model_id, label), image)
    sess.close()


