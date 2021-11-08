import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from time import time

from .algorithm import Algorithm


class Marques(Algorithm):
    ''' Algorithm Marques 2013. '''

    def __init__(self, radius_search=0, verbose=False):
        
        self.radius_search = 10
        self.inference_time = 0 # actually, inference means here distance computation
        self.verbose = verbose

        # result 
        self.compatibilities = None
   

    def _compute_matrix(self, strips, d):
        ''' Compute cost matrix. '''

        # minimum height (size of intersection)
        min_h = min([strip.h for strip in strips.strips])
        
        # features extraction
        features = []
        for strip in strips.strips:
            features.append(self._extract_features(strip, d, min_h))
        l, r = list(zip(*features))

        # distance computation
        R = self.radius_search
        N = len(strips.strips)
        matrix = np.zeros((N, N), dtype=np.float32)
        window_size = min_h - 2 * R
        self.inference_time = 0
        pair = 1
        total_pairs = N * (N - 1)
        for i in range(N):
            for j in range(N):
                if i != j:
                    t0 = time()
                    min_dist = float('inf')
                    for k in range(2 * R + 1): # slide displacement
                        dist = euclidean(r[i][R : R + window_size], l[j][k : k + window_size])
                        if dist < min_dist:
                            min_dist = dist
                    matrix[i, j] = min_dist
                    self.inference_time += (time() - t0)
                    if self.verbose and (pair % 500 == 0):
                        remaining = self.inference_time * (total_pairs - pair) / pair
                        print('[{:.2f}%] pair={}/{} :: elapsed={:.2f}s predicted={:.2f}s pair time={:.3f}s '.format(
                            100 * pair / total_pairs, pair, total_pairs, self.inference_time,
                            remaining, self.inference_time / pair
                        ))
                    pair += 1

        np.fill_diagonal(matrix, 1e7)
        return matrix

    
    def _extract_features(self, strip, d, size):
        ''' Features. '''
    
        # value channel
        V = cv2.cvtColor(strip.image, cv2.COLOR_RGB2HSV)[:, :, 2]

        # borders
        l = strip.offsets_l[: size] + d
        r = strip.offsets_r[: size] - d

        # features
        idx = np.arange(size)
        left = V[idx, l + d]
        right = V[idx, r - d]
        return left, right
                
    
    def run(self, strips, d=3):
        ''' Run algorithm. '''
        
        self.compatibilities = self._compute_matrix(strips, d)
        return self

    
    def name(self):
        
        return 'marques'
