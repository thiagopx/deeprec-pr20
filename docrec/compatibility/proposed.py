import sys
from time import time
import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
import numpy as np
import math
import tensorflow as tf

from .algorithm import Algorithm
from ..neural.models.squeezenet import SqueezeNet
from ..neural.models.squeezenet_simple_bypass import SqueezeNetSB


class Proposed(Algorithm):
    '''  Proposed algorithm. '''

    def __init__(
        self, arch, weights_path, vshift, input_size, num_classes,
        thresh_method='sauvola', verbose=False, seed=None, offset=None
    ):

        assert arch in ['sn', 'sn-bypass']
        assert thresh_method in ['otsu', 'sauvola']

        # preparing model
        self.offset = offset
        self.vshift = vshift
        self.input_size_h, self.input_size_w = input_size
        self.images_ph = tf.placeholder(
            tf.float32, name='images_ph', shape=(None, self.input_size_h, self.input_size_w, 3) # channels last
        )
        self.batch = np.ones((2 * vshift + 1, self.input_size_h, self.input_size_w, 3), dtype=np.float32)

        # model
        if arch == 'sn':
            model = SqueezeNet(self.images_ph, num_classes=num_classes, mode='test', channels_first=False)
        else:
            model = SqueezeNetSB(self.images_ph, num_classes=num_classes, mode='test', channels_first=False)
        logits = model.output
        probs = tf.nn.softmax(logits)
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

        # vertical offset
        offset = (h - self.input_size_h) // 2 if self.offset is None else self.offset

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
