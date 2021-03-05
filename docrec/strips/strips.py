import os
import re
import numpy as np
import cv2
from random import shuffle
import matplotlib.pyplot as plt
from skimage import transform

from .strip import Strip


class Strips(object):
    ''' Strips operations manager.'''

    def __init__(self, path=None, strips_list=None, filter_blanks=True, blank_tresh=127):
        ''' Strips constructor.

        @path: path to strips (in case of load real strips)
        @strips_list: list of strips (objects of Strip class)
        @filter_blanks: true-or-false flag indicating the removal of blank strips
        @blank_thresh: threshold used in the blank strips filtering
        '''

        assert (path is not None) or (strips_list is not None)

        self.strips = []
        self.artificial_mask = False
        if path is not None:
            assert os.path.exists(path)
            self._load_data(path)
        else:
            self.strips = [strip.copy() for strip in strips_list]

        if filter_blanks:
            self.strips = [strip for strip in self.strips if not strip.is_blank(blank_tresh)]


    def __call__(self, i):
        ''' Returns the i-th strip. '''

        return self.strips[i]


    def _load_data(self, path, regex_str='.*\d\d\d\d\d\.*'):
        ''' Stack strips horizontally.

        Strips are images with same basename (and extension) placed in a common
        directory. Example:

        basename="D001" and extension=".jpg" => strips D00101.jpg, ..., D00130.jpg.
        '''

        path_images = '{}/strips'.format(path)
        path_masks = '{}/masks'.format(path)
        regex = re.compile(regex_str)

        # loading images
        fnames = sorted([fname for fname in os.listdir(path_images) if regex.match(fname)])
        images = []
        for fname in fnames:
            image = cv2.cvtColor(
                cv2.imread('{}/{}'.format(path_images, fname)),
                cv2.COLOR_BGR2RGB
            )
            images.append(image)

        # load masks
        masks = []
        if os.path.exists(path_masks):
            for fname in fnames:
                mask = np.load('{}/{}.npy'.format(path_masks, os.path.splitext(fname)[0]))
                masks.append(mask)
        else:
            masks = len(images) * [None]
            self.artificial_mask = True

        for position, (image, mask) in enumerate(zip(images, masks), 1):
            strip = Strip(image, position, mask)
            self.strips.append(strip)


    def trim(self, left=0, right=0):
        ''' Trim borders from strips. '''

        n = len(self.strips)
        self.strips = self.strips[left : n - right]
        return self


    def image(self, order=None, displacements=None, filled=False, verbose=False):
        ''' Return the reconstruction image in a specific order . '''

        N = len(self.strips)
        if order is None:
            order = list(range(N))
        if displacements is None:
            displacements = N * [0]
        prev = order[0]
        result = self.strips[prev].copy()
        i = 1
        total = len(order) - 1
        for curr, disp in zip(order[1 :], displacements):
            if verbose:
                 print('stacking strip {}/{}'.format(i, total))
            i += 1
            result.stack(self.strips[curr], disp=disp, filled=filled)
        return result.image


    def pair(self, i, j, filled=False, accurate=False):
        ''' Return a single image with two paired strips. '''

        if accurate:
            return self._align(i, j) # filled not used

        return self.strips[i].copy().stack(self.strips[j], filled).image


    def plot(self, size=(8, 8), fontsize=6, ax=None, show_lines=False):
        ''' Plot strips given the current order. '''

        assert len(self.strips) > 0
        if ax is None:
            fig = plt.figure(figsize=size, dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig = None

        shapes = [[strip.h, strip.w] for strip in self.strips]
        max_h, max_w = np.max(shapes, axis=0)
        sum_h, sum_w = np.sum(shapes, axis=0)

        # Background
        offsets = [0]
        background = self.strips[0].copy()
        for strip in self.strips[1 :]:
            offset = background.stack(strip).image.shape[1]
            print(offset, strip.image.shape[0])
            offsets.append(offset)

        ax.imshow(background.image)
        ax.axis('off')

        for strip, offset in zip(self.strips, offsets):
            d = strip.w / 2
            ax.text(
                offset + d, 50, str(strip.position), color='blue',
                fontsize=fontsize, horizontalalignment='center'
            )
        if show_lines:
            ax.vlines(
                offsets[1 :], 0, max_h, linestyles='dotted', color='red',
                linewidth=0.5
            )
        return fig, ax, offsets