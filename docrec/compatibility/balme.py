import cv2
import numpy as np

from .algorithm import Algorithm


class Balme(Algorithm):
    '''
    Algorithm Balme

    Balme, J.: Reconstruction of shredded documents in the absence of shape
    information (2007) Working paper, Dept. of Computer Science, Yale
    University, USA.
    '''

    def __init__(self, tau=0.1):

        self.tau = tau
        
        # result 
        self.compatibilities = None
        

    def _compute_matrix(self, strips, d):
        ''' Compute cost matrix. '''

        # distance computation: Gaussian correlation
        dist = lambda x, y: np.sum(
            np.correlate(
                np.logical_xor(x, y), [0.05, 0.1, 0.7, 0.1, 0.05]
            ) > self.tau
        )
        
        min_h = min([strip.h for strip in strips.strips])

        features = []
        for strip in strips.strips:
            features.append(self._extract_features(strip, d, min_h))

        N = len(strips.strips)
        matrix = np.zeros((N, N), dtype=np.float32)
        l, r = list(zip(*features))
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i, j] = dist(r[i], l[j])

        np.fill_diagonal(matrix, 1e7)
        return matrix
    
    
    def _extract_features(self, strip, d, size):
        ''' Features. '''
    
        # inverted thresholded image
        _, image_bin = cv2.threshold(
            cv2.cvtColor(strip.image, cv2.COLOR_RGB2GRAY),
            0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
                
        # borders
        l = strip.offsets_l[: size] + d
        r = strip.offsets_r[: size] - d

        # features
        idx = np.arange(size)
        left = image_bin[idx, l + d]
        right = image_bin[idx, r - d]
        return left, right
    
    
    def run(self, strips, d=0):
        
        self.compatibilities = self._compute_matrix(strips, d)
        return self
    
    
    def name(self):

        return 'balme'
