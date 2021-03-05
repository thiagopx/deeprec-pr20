import os
import numpy as np
import localsolver


class SolverLS:

    def __init__(self, maximize=False):

        self.solution = None
        self.maximize = maximize


    def solve(self, instance, time_limit=5):

        instance = np.array(instance)
        if self.maximize:
            np.fill_diagonal(instance, 0)
            instance = instance.max() - instance # transformation function (similarity -> distance)

        instance = np.pad(
            instance, ((0, 1), (0, 1)), mode='constant', constant_values=0 # dummy node
        )
        np.fill_diagonal(instance, 1e7)#instance.max()) # self loops
        num_cities = instance.shape[0]

        with localsolver.LocalSolver() as ls:
            # model
            model = ls.model
            cities = model.list(num_cities)
            model.constraint(model.count(cities) == num_cities)
            distance_array = model.array(instance.tolist())

            # minimize the total distance
            dist_selector = model.function(lambda i: model.at(distance_array, cities[i - 1], cities[i]))
            obj = (model.sum(model.range(1, num_cities), dist_selector)
                + model.at(distance_array, cities[num_cities - 1], cities[0]))
            model.minimize(obj)
            model.close()

            # time limit
            ls.create_phase().time_limit = time_limit
            ls.solve()
            solution = [c for c in cities.value]

        dummy_idx = solution.index(num_cities - 1)
        self.solution = solution[dummy_idx + 1 : ] + solution[: dummy_idx]
        return self
