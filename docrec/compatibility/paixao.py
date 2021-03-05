import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import squareform
from time import time
import sys

from .algorithm import Algorithm
from docrec.image.processing.transform import distance
from docrec.clustering.kmedoids.kmedoids import KMedoids


# def MHD(A, B):
#     '''
#     Adapted from https://github.com/sapphire008/Python/blob/master/generic/HausdorffDistance.py

#     M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
#     matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
#     http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
#     '''
#     #t0 = time()
#     #D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B)- 2*(np.dot(A, B.T))) # pairwise distance
#     D_mat = np.sqrt(- 2 * np.dot(A, B.T) + np.sum(A ** 2, axis=1)[:, np.newaxis] + np.sum(B ** 2, axis=1)) # pairwise distance
#     FHD = np.mean(np.min(D_mat, axis=1)) # forward
#     RHD = np.mean(np.min(D_mat, axis=0)) # reverse
#     MHD = np.max(np.array([FHD, RHD]))
#     #print(time() - t0)
#     return MHD


class Paixao(Algorithm):
    '''  Paixao algorithm. '''

    def __init__(
        self, FFm=1.0, CC=1.0, EC=0.0, p=-0.2, gamma=0.0125,
        threshold=1.0, ns=10, max_cache_size=10000000, seed=0, verbose=True,
        trailing=0
    ):

        self.verbose = verbose
        self.trailing = trailing

        self.strips = None

        # random number generator
        self.rng = np.random.RandomState(seed)

        # inner distances (for clustering)
        self.inner_distances = None

        # distance transform cache structure
        self.window = None
        self.cache_distance_transform = None
        self.coords = None

        # cache configuration
        self.max_cache_size = max_cache_size
        self.cache_distances = None

        # results
        self.representative = None
        self.matrix = None

        # parameters
        self.FFm = FFm
        self.EC = EC
        self.CC = CC
        self.FF = p
        self.FC = p
        self.EF = p
        self.gamma = gamma
        self.threshold = threshold
        self.ns = ns


    def _cache_distance_transform(self):
        ''' Caching distance transform. '''

        assert self.window is not None

        for char in self.strips.all:
            id_ = id(char)
            self.cache_distance_transform[id_] = distance(char, window=self.window)


    def _distance(self, char1, char2):
        ''' Distance (dissimilarity) between two characters. '''

        id1 = id(char1)
        id2 = id(char2)
        try:
            dist = self.cache_distances[(id1, id2)]
        except KeyError:
            distance_transform1, base1 = self.cache_distance_transform[id1]
            distance_transform2, base2 = self.cache_distance_transform[id2]
            h12 = distance_transform1[base2 == 255].mean()
            h21 = distance_transform2[base1 == 255].mean()
            dist = max(h12, h21)
            if len(self.cache_distances) < self.max_cache_size:
                self.cache_distances[(id1, id2)] = dist
        return dist


    # def _distance(self, char1, char2):
    #     ''' Distance (dissimilarity) between two characters. '''

    #     id1 = id(char1)
    #     id2 = id(char2)
    #     try:
    #         dist = self.cache_distances[(id1, id2)]
    #     except KeyError:
    #         try:
    #             distance_transform1, base1 = self.cache_distance_transform[id1]
    #         except KeyError:
    #             distance_transform1, base1 = distance(char1, window=self.window)
    #         distance_transform2, base2 = self.cache_distance_transform[id2]
    #         h12 = distance_transform1[base2 == 255].mean()
    #         h21 = distance_transform2[base1 == 255].mean()
    #         dist = max(h12, h21)
    #         if len(self.cache_distances) < self.max_cache_size:
    #             self.cache_distances[(id1, id2)] = dist
    #     return dist


    # def _distance(self, char1, char2):
    #     ''' Distance (dissimilarity) between two characters. '''

    #     id1 = id(char1)
    #     id2 = id(char2)
    #     try:
    #         dist = self.cache_distances[(id1, id2)]
    #     except KeyError:
    #         dist = MHD(self.coords[id1], self.coords[id2])
    #         if len(self.cache_distances) < self.max_cache_size:
    #             self.cache_distances[(id1, id2)] = dist
    #     return dist


    def _compute_inner_distances(self):
        ''' Inner characters distantes for clustering. '''

        _, chars = zip(*self.strips.inner)

        # Compute distances
        dists = [self._distance(char1, char2) for i, char1 in enumerate(chars[: -1]) for char2 in chars[i + 1 :]]
        self.inner_distances = squareform(dists)

    
    # def _compute_inner_distances(self):
    #     ''' Inner characters distantes for clustering. '''

    #     _, chars = zip(*self.strips.inner)

    #     n = len(chars)
    #     total = (n * (n - 1)) // 2
    #     k = 0
    #     # Compute distances
    #     dists = []
    #     t0 = time()
    #     for i, char1 in enumerate(chars[: -1]):
    #         for char2 in chars[i + 1 :]:
    #             dists.append(self._distance(char1, char2))
    #             k += 1
    #             if k % 1000 == 0:
    #                 mean_time = (time() - t0) / k
    #                 estimated = (total - k) * mean_time
    #                 print('[{}/{}] - {:.2f}%] - Estimated {:.2f}s'.format(k, total, 100*k/total, estimated))
    #     self.inner_distances = squareform(dists)


    def _is_character(self, char):
        ''' Search for any similar character. '''

        for other in self.representative:
            if self._distance(char, other) <= self.threshold:
                return True
        return False


    def _score_tresh(self, charr, charl, joined):
        ''' Threshold-based score function. '''

        # EC EF
        if charr is None:
            if self._is_character(charl):
                return self.EC
            return self.EF
        elif charl is None:
            if self._is_character(charr):
                return self.EC
            return self.EF

        # FFm
        if self._is_character(joined):
            return self.FFm

        # CC CF FC FF
        rtype = 'C' if self._is_character(charr) else 'F'
        ltype = 'C' if self._is_character(charl) else 'F'
        mtype =  rtype + ltype
        if mtype == 'CC':
            return self.CC
        if mtype == 'FF':
            return self.FF
        return self.FC


    def _compute_score(self, seed):
        ''' Compute pairwise score. '''

        # Clustering
        # Obs: parameter k value not used until this moment.
        X = self.inner_distances
        n = X.shape[0]

        #k = n / 4
        # ISO basic Latin alphabet
        k = min(n, 52)
        max_neighbor = int(self.gamma * n * (n - k))
        kmedoids = KMedoids(
            seed=seed, init='random', num_local=2, max_neighbor=max_neighbor
        ).run(X, k)

        # Representative characters
        _, inner = zip(*self.strips.inner)
        self.representative = [inner[i] for i in kmedoids.medoids]
        score_func = self._score_tresh
        score = {}
        for pair in self.strips.matchings:
            val = 0
            for charr, charl, joined in self.strips.matchings[pair]:
                val += score_func(charr, charl, joined)
            score[pair] = val
        return score


    def _compute_matrix(self, seed):
        ''' Compute matrix. '''

        assert self.inner_distances is not None
        assert self.FFm is not None
        assert self.EC is not None
        assert self.FF is not None
        assert self.CC is not None
        assert self.EF is not None
        assert self.gamma is not None
        assert self.threshold is not None

        # Score computation
        score = self._compute_score(seed)

        # Filling cost matrix
        n = len(self.strips.strips)
        matrix = np.zeros((n, n), dtype=np.float32)
        for pair in score:
            matrix[pair] = score[pair]

        # Transformation function
        #matrix = matrix.max() - matrix
        #np.fill_diagonal(matrix, np.inf)
        return matrix


    # def _convert2coords(self):

    #     for char in self.strips.all:
    #         id_ = id(char)

    #         # (x, y) contour representation 
    #         _, contours, hierarchy = cv2.findContours(char, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #         filtered = []
    #         idx = 0
    #         while idx >= 0:
    #             filtered.append(contours[idx].squeeze())
    #             idx = hierarchy[0][idx][0]
    
    #         #coords = np.vstack([cnt.squeeze() for cnt in contours]).astype(np.int16)
    #         coords = np.vstack(filtered).astype(np.int16)
    #         centroid = np.mean(np.transpose(np.where(char == 255)), axis=0).astype(np.int16)[::-1] # (y, x) -> (x, y)
    #         self.coords[id_] = coords - centroid # center on centroid


    def run(self, strips):
        ''' Run algorithm. '''

        self.strips = strips
        max_w = max([char.shape[1] for char in self.strips.all])
        max_h = max([char.shape[0] for char in self.strips.all])
        self.window= (2 * max_w, 2 * max_h)

        self.cache_distance_transform = {}
        if self.verbose:
            print('{}Caching distance transform... '.format(self.trailing * ' '), end='')
            sys.stdout.flush()
            t0 = time()
        self._cache_distance_transform()
        if self.verbose:
            print('Elapsed time: {:.2f}s'.format(time() - t0))
            sys.stdout.flush()

        # self.coords = {}
        # if self.verbose:
        #     print('{}Converting to coordinates... '.format(self.trailing * ' '), end='')
        #     sys.stdout.flush()
        #     t0 = time()
        # self._convert2coords()
        # if self.verbose:
        #     print('Elapsed time: {:.2f}s'.format(time() - t0))
        #     sys.stdout.flush()

        self.inner_distances = None
        self.cache_distances = {}
        if self.verbose:
            print('{}Computing inner distances... '.format(self.trailing * ' '), end='')
            sys.stdout.flush()
            t0 = time()
        self._compute_inner_distances()
        if self.verbose:
            print('Elapsed time: {:.2f}s'.format(time() - t0))
            sys.stdout.flush()

        # matrix computation
        matrices = []
        for s in range(1, self.ns + 1):
            if self.verbose:
                print('{}solution {} - (cache size={})... '.format(
                    self.trailing * ' ', s, len(self.cache_distances)
                ), end='')
                sys.stdout.flush()
                t0 = time()
            # seed for clustering
            seed_ = self.rng.randint(0, 4294967295)
            matrix = self._compute_matrix(seed_)
            matrices.append(matrix)
            if self.verbose:
                print('Elapsed time: {:.2f}s'.format(time() - t0))
                sys.stdout.flush()
        self.matrix = np.stack(matrices)
        return self


    def name(self):
        ''' Method name. '''

        return 'paixao'


# Caching distance transform... Elapsed time: 36.81s
# Computing inner distances... Elapsed time: 20.64s
# solution 1 - (cache size=386760)... Elapsed time: 52.27s
# solution 2 - (cache size=1100999)... Elapsed time: 22.82s
# solution 3 - (cache size=1291235)... Elapsed time: 16.48s
# solution 4 - (cache size=1369797)... Elapsed time: 14.93s
# solution 5 - (cache size=1406089)... Elapsed time: 15.76s
# solution 6 - (cache size=1467445)... Elapsed time: 17.26s
# solution 7 - (cache size=1557216)... Elapsed time: 17.32s
# solution 8 - (cache size=1631996)... Elapsed time: 14.36s
# solution 9 - (cache size=1675821)... Elapsed time: 17.87s
# solution 10 - (cache size=1763414)... Elapsed time: 15.08s

# Computing inner distances... Elapsed time: 492.37s
# solution 1 - (cache size=386760)... Elapsed time: 870.88s
# solution 2 - (cache size=1100999)... Elapsed time: 270.56s
# solution 3 - (cache size=1291235)... Elapsed time: 116.24s
# solution 4 - (cache size=1369797)... Elapsed time: 58.47s
# solution 5 - (cache size=1406089)... Elapsed time: 87.40s
# solution 6 - (cache size=1467445)... Elapsed time: 110.35s
