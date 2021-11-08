
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
import tensorflow as tf

from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips
from docrec.neural.models.squeezenet import SqueezeNet
from docrec.neural.models.reduced_squeezenet import RedSqueezeNet
from docrec.ndarray.utils import first_nonzero, last_nonzero


# def extract_features(strip, input_size, thresh_method='sauvola'):
#     ''' Extract image around the border. '''

#     image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
#     thresh_func = threshold_sauvola if thresh_method == 'sauvola' else threshold_otsu
#     thresh = thresh_func(image)
#     thresholded = (image > thresh).astype(np.float32)

#     # _, thresh = cv2.threshold(
#     #     cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     # )
#     image_bin = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last

#     wl = math.ceil(input_size_w / 2)
#     wr = int(input_size_w / 2)
#     h, w, _ = strip.image.shape
#     offset = int((h - input_size_h) / 2)

#     # left image
#     left_border = strip.offsets_l
#     left = np.ones((input_size_h, wl, 3), dtype=np.float32)
#     for y, x in enumerate(left_border[offset : offset + input_size_h]):
#         w_new = min(wl, w - x)
#         left[y, : w_new] = image_bin[y + offset, x : x + w_new]

#     # right image
#     right_border = strip.offsets_r
#     right = np.ones((input_size_h, wr, 3), dtype=np.float32)
#     for y, x in enumerate(right_border[offset : offset + input_size_h]):
#         w_new = min(wr, x + 1)
#         right[y, : w_new] = image_bin[y + offset, x - w_new + 1: x + 1]

#     return left, right


def create_overlay(image, conv10, input_size, view_mode):

    # data
    # input_size = tuple(args.input_size)
    input_size_h, input_size_w = input_size
    wr = input_size_w // 2
    neg = conv10[:, :, 0]
    pos = conv10[:, :, 1]
    neu = conv10[:, :, 2]

    if view_mode == 'norm':
        pos = pos / (pos + neg + 1e-5)
        neg = neg / (pos + neg + 1e-5)
    
    maps = np.stack([neu, pos, neg]) # red or green (BGR)

    if view_mode == 'global':
        maps = maps / maps.max()
    
    maps = (255 * np.transpose(maps, axes=(1, 2, 0))).astype(np.uint8)
    maps = cv2.resize(maps, dsize=(input_size_w, input_size_h), interpolation=cv2.INTER_CUBIC)

    maps_max = maps.copy()
    maps_max[maps != maps.max(axis=2, keepdims=True)] = 0
    return maps, maps_max


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
    # image = np.ones((input_size_h, input_size_w, 3))
    wr = input_size_w // 2
    images_ph = tf.placeholder(
        tf.float32, name='images_ph', shape=(None, input_size_h, input_size_w, 3) # channels last
    )
    
    # 1% black pixels
    black_prob = 0.01
    batch = np.random.rand(input_size_h, input_size_w)
    batch = np.stack(3 * [batch])
    batch= batch.transpose((1, 2, 0)).copy()
    batch = batch[np.newaxis]
    batch[0, : 50] = 0
    batch[0,  50 : 60] = 1
    batch[0,  60 : 70] = 0
    batch[0,  70 : 90] = 1
    batch[0,  90 : 110] = 0
    batch[0, 100 :] = (batch[0, 100 :] >= black_prob)
    print(np.unique(batch))
    batch = (255 * batch).astype(np.uint8)
    print(np.unique(batch))
    print(batch.shape)
    # horizontal line
    # batch[0, 200] = 0
    # vertical line
    # batch[0, :, 200] = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    print(batch[0].shape)
    cv2.putText(batch[0], 'AaBbCcDdEeFfGgHhJj', (10, 300), font, 0.5, (0, 0, 0), 2, 8)
    cv2.putText(batch[0], 'AaBbCcDdEeFfGgHhJj', (10, 400), font, 1, (0, 0, 0), 2, 8)
    cv2.putText(batch[0], 'AaBbCcDdEeFfGgHhJj', (10, 500), font, 2, (0, 0, 0), 2, 8)
    print(np.unique(batch))
    batch = batch / 255.0
    print(np.unique(batch))

    # batch[0] = image
        
    # model
    if args.arch == 'sn':
        model = SqueezeNet(images_ph, num_classes=3, mode='test', channels_first=False)
    else:
        model = RedSqueezeNet(images_ph, num_classes=3, mode='test', channels_first=False)

    sess = model.sess
    # logits_op = model.output[:, : -1] # exclude neutral
    logits_op = model.output
    probs_op = logits_op / (tf.reduce_sum(logits_op, axis=1, keepdims=True) + 1e-5)
    conv10_op = model.view

    base_path = 'illustration/bias'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    sess.run(tf.global_variables_initializer())
    # model_id = args.model_id if args.model_id is not None else '{}-{}'.format(training_set, args.arch)
    for val in [0.0]:#, 0.05, 0.25, 0.5, 1.0]:#'isri-ocr_0.0_', 'isri-ocr-0.3']:#cdip-sn-rneu=0.0', 'cdip-sn-rneu=0.05', 'isri-ocr-sn-rneu=0.0', 'isri-ocr-sn-rneu=0.05']:
        model_id = 'cdip_test'
        # model_id = 'isri-ocr_{}_1000_32x32'.format(val)
        weights_path = json.load(open('traindata/{}/info.json'.format(model_id), 'r'))['best_model']
        #weights_path = 'traindata/isri-ocr_test/model/1.npy'
        model.load_weights(weights_path)
        probs, conv10 = sess.run([probs_op, conv10_op], feed_dict={images_ph: batch})
        # print('{} (-)={:5.2f}% (+)={:5.2f}% (N)={:5.2}%'.format(model_id, 100 * probs[0, 0], 100 * probs[0, 1], 100 * probs[0, 2]))
        print('{} (-)={:5.2f}% (+)={:5.2f}% (n.)={:5.2f}%'.format(model_id, 100 * probs[0, 0], 100 * probs[0, 1], 100 * probs[0, 2]))
        overlay_mixed, overlay_max = create_overlay((255 * batch[0]).astype(np.uint8), conv10[0], input_size, args.view_mode)
        # print(overlay)
        image = cv2.addWeighted(overlay_mixed, args.alpha, (255 * batch[0]).astype(np.uint8), 1 - args.alpha, 0)
        cv2.imwrite('{}/{}-mixed.jpg'.format(base_path, model_id), image)
        image = cv2.addWeighted(overlay_max, args.alpha, (255 * batch[0]).astype(np.uint8), 1 - args.alpha, 0)
        cv2.imwrite('{}/{}-max.jpg'.format(base_path, model_id), image)


    sess.close()


