from itertools import combinations
import numpy as np

from .algorithms.pam import pam
from .algorithms.clarans import clarans


class KMedoids:
    ''' K-Medoids clustering.

    References:
    [1] http://www.mathworks.com/help/stats/kmedoids.html
    [2] https://en.wikipedia.org/wiki/K-medoids
    '''

    def __init__(
        self, algorithm='clarans', init='random', seed=None, verbose=False,
        **kwargs
    ):

        assert algorithm in ['pam', 'clarans']

        # general parameters
        self.algorithm = algorithm
        self.init = init
        self.seed = seed
        self.verbose = verbose

        # algorithm functions
        self.functions = {'pam': pam, 'clarans': clarans}

        # custom parameters
        self.custom_params = \
            {'pam': {'max_it': 100},
             'clarans': {'num_local': 2, 'max_neighbor': 200}
            }

        # adjusting parameters
        for key, value in kwargs.items():
            if key in self.custom_params[algorithm]:
                self.custom_params[algorithm][key] = value
            else:
                raise Exception('%s: parameter %s invalid for algorihtm %s.'
                                % (self.__init__.__name__, key, algorithm))

        self.clusters = None
        self.medoids = None
        self.cost = None
        self.X = None


    def run(self, X, k):
        ''' Run clustering. '''

        function = self.functions[self.algorithm]
        custom_params = self.custom_params[self.algorithm]

        # run kmedoids function
        self.clusters, self.cost = function(
            X, k, init=self.init, seed=self.seed, verbose=self.verbose,
            **custom_params
        )

        filtered = []
        for cluster in self.clusters:
           if len(cluster) >= 2:
              filtered.append(cluster)
        if filtered:
            self.clusters = filtered
        self.medoids = [cluster[0] for cluster in self.clusters]
        return self

