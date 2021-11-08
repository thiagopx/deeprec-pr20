import os
import numpy as np


class SolverNN:

    def __init__(self, maximize=False, mode='repetitive', start=0, inverse=False, seed=None):

        self.solution = None
        self.cost = 0
        self.maximize = maximize
        self.mode = mode
        self.start = start
        self.inverse = inverse
        self.seed = seed


    def solve(self, instance):
    
        assert self.mode in ('random', 'fixed', 'repetitive')

        instance = np.array(instance)
        if self.maximize:
            np.fill_diagonal(instance, 0)
            instance = instance.max() - instance # transformation function (similarity -> distance)
        num_cities = instance.shape[0]
        matrix = np.ma.array(instance, mask=False)
   
        # Start vertex (random, all, fixed)
        if self.mode == 'repetitive':
            start = list(range(num_cities))
        elif self.mode == 'random':
            if self.seed is not None:
                np.random.seed(self.seed)
            start = [np.random.randint(0, num_cities)]
        else:
            start = [start]
   
        path = []
        insert = lambda v: path.insert(0, v) if self.inverse else path.append(v)
        visit = lambda v: (
            matrix.mask[v, :] if self.inverse else matrix.mask[:, v]
        ).fill(True)
        current = lambda: path[0] if self.inverse else path[-1]
        best_neighbor = lambda v: int(
            (matrix[:, v] if self.inverse else matrix[v, :]).argmin()
        )
        get_cost = lambda v, w: matrix[w, v] if self.inverse else matrix[v, w]
    
        # Search best solution for each start vextex
        min_cost = float('inf')
        best = None
        for v in start:
            path[:] = [v]
            visit(v)
            cost = 0
            for i in range(num_cities - 1):
                v = current()
                w = best_neighbor(v)
                cost += get_cost(v, w)
                insert(w)
                visit(w)
            matrix.mask.fill(False)

            # Check if the current path is the best one
            if cost < min_cost:
                min_cost = cost
                best = list(path)
    
        self.solution = best
        self.cost = min_cost
        return self

