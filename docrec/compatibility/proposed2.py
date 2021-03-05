from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from time import time
import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
import numpy as np
import math
import tensorflow as tf
from keras import backend as K

from .algorithm import Algorithm
from ..neural.models.squeezenet import SqueezeNet


class Proposed(Algorithm):
    '''  Proposed algorithm. '''

    def __init__(
        self, arch, weights_path, vshift, input_size, num_classes, 
        thresh_method='sauvola', neutral_thresh=0.2, verbose=False, seed=None
    ):

        assert arch in ['sn']
        assert thresh_method in ['otsu', 'sauvola']

        # preparing model
        self.vshift = vshift
        self.input_size_h, self.input_size_w = input_size
        self.images_ph = tf.placeholder(
            tf.float32, name='images_ph', shape=(None, self.input_size_h, self.input_size_w, 3) # channels last
        )
        self.batch = np.ones((2 * vshift + 1, self.input_size_h, self.input_size_w, 3), dtype=np.float32)

        # model
        model = SqueezeNet(self.images_ph, num_classes=num_classes, mode='test', channels_first=False)
        logits = model.output
        
        conv10 = model.view # (batch, feat_height, 1, #classes=2)
        feat_height = conv10.get_shape().as_list()[1]
        ksizes = [1, self.input_size_w, self.input_size_w, 1]
        strides = [1, self.input_size_h // feat_height, 1, 1]
        patches = tf.extract_image_patches(
            self.images_ph, ksizes, strides, rates=[1, 1, 1, 1], padding='VALID' # (batch, feat_height, 1, #pixels=32*32*3) 
        )
        # excluding neutral samples
        self.mask = tf.cast(tf.reduce_mean(patches, axis=3) <= 1 - neutral_thresh, tf.float32) # (batch, feat_height, 1)
        # print(mask.get_shape())
        # weight = 0.65
        # neg = conv10[..., 0] # (batch, feat_height, 1)
        # pos = conv10[..., 1] # (batch, feat_height, 1)
        # pos = mask * pos + (1 - mask) * neg
        # neg = mask * neg + (1 - mask) * pos

        # # masked_conv10 = mask[..., None] * conv10
        # # self.neutral = tf.nn.softmax(tf.reduce_sum((1.0 - mask[..., None]) * conv10, axis=1))
        # # # masked_conv10 = mask[..., None] * conv10 + weight * (1.0 - mask[..., None]) * conv10
        # masked_conv10 = tf.stack([neg, pos], axis=3)
        # logits = tf.reduce_mean(masked_conv10, (1, 2), keepdims=False)
        probs = tf.nn.softmax(logits)
        # probs = logits / (tf.reduce_sum(logits, axis=1, keepdims=True) + 1e-05)

        self.comp_op = tf.reduce_max(probs[:, 1])
        self.disp_op = tf.argmax(probs[:, 1]) - vshift

        # result
        self.compatibilities = None
        self.displacements = None

        # init model
        self.sess = model.sess
        self.sess.run(tf.global_variables_initializer())
        model.load_weights(weights_path)

        self.verbose = verbose
        self.inferente_time = 0
        self.thresh_method = thresh_method


    def _extract_features(self, strip):
        ''' Extract image around the border. '''

        image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
        thresh_func = threshold_sauvola if self.thresh_method == 'sauvola' else threshold_otsu
        thresh = thresh_func(image)
        thresholded = (image > thresh).astype(np.float32)
        image_bin = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last

        wl = math.ceil(self.input_size_w / 2)
        wr = int(self.input_size_w / 2)
        h, w, _ = strip.image.shape
        offset = int((h - self.input_size_h) / 2)

        # left image
        left_border = strip.offsets_l
        left = np.ones((self.input_size_h, wl, 3), dtype=np.float32)
        for y, x in enumerate(left_border[offset : offset + self.input_size_h]):
            w_new = min(wl, w - x)
            left[y, : w_new] = image_bin[y + offset, x : x + w_new]

        # right image
        right_border = strip.offsets_r
        right = np.ones((self.input_size_h, wr, 3), dtype=np.float32)
        for y, x in enumerate(right_border[offset : offset + self.input_size_h]):
            w_new = min(wr, x + 1)
            right[y, : w_new] = image_bin[y + offset, x - w_new + 1: x + 1]

        return left, right


    def run(self, strips, d=0, ignore_pairs=[]): # d is not used at this moment
        ''' Run algorithm. '''

        N = len(strips.strips)
        compatibilities = np.zeros((N, N), dtype=np.float32)
        displacements = np.zeros((N, N), dtype=np.int32)
        wr = int(self.input_size_w / 2)

        # features
        features = []
        for strip in strips.strips:
            left, right = self._extract_features(strip)
            features.append((left, right))

        inference = 1
        self.inference_time = 0
        total_inferences = N * (N - 1)
        for i in range(N):
            self.batch[:, :, : wr] = features[i][1]

            for j in range(N):
                if i == j or (i, j) in ignore_pairs:
                    continue

                feat_j = features[j][0]
                self.batch[self.vshift, :, wr : ] = feat_j
                for r in range(1, self.vshift + 1):
                    self.batch[self.vshift - r, : -r, wr :] = feat_j[r :]  # slide up
                    self.batch[self.vshift + r, r : , wr :] = feat_j[: -r] # slide down

                t0 = time()
                comp, disp = self.sess.run([self.comp_op, self.disp_op], feed_dict={self.images_ph: self.batch})
                self.inference_time += (time() - t0)
                if self.verbose and (inference % 20 == 0):
                    remaining = self.inference_time * (total_inferences - inference) / inference
                    print('[{:.2f}%] inference={}/{} :: elapsed={:.2f}s predicted={:.2f}s mean inf. time={:.3f}s '.format(
                        100 * inference / total_inferences, inference, total_inferences, self.inference_time,
                        remaining, self.inference_time / inference
                    ))
                inference += 1

                compatibilities[i, j] = comp
                displacements[i, j] = disp

        self.compatibilities = compatibilities
        self.displacements = displacements
        return self


    def name(self):
        ''' Method name. '''

        return 'proposed'
