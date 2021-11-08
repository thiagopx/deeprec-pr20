
import os
import numpy as np


class SolverKBH:

    def __init__(self, maximize=False):
        ''' Kruskal-based solver.
        
        Adapted from Algorithms-Book--Python
        Python implementations of the book "Algorithms - Dasgupta, Papadimitriou and Vazurani"
        https://github.com/israelst/Algorithms-Book--Python
        '''

        self.solution = None
        self.cost = 0
        self.maximize = maximize
        self.parent = dict()


    def make_set(self, vertice):

        self.parent[vertice] = vertice


    def find(self, vertice):

        if self.parent[vertice] != vertice:
            self.parent[vertice] = self.find(self.parent[vertice])
        return self.parent[vertice]


    def union(self, vertice1, vertice2, vertices):
        root1 = self.find(vertice1)
        self.parent[vertice2] = root1
        for vertice in vertices:
            self.find(vertice)


    def solve(self, instance):

        matrix = np.array(instance)
        if self.maximize:
            np.fill_diagonal(matrix, 0)
            matrix = matrix.max() - matrix # transformation function (similarity -> distance)
        N = matrix.shape[0]
       
        graph = {
            'vertices': [v for v in range(N)],
            'edges': set([(matrix[u, v], u, v) for u in range(N) for v in range(N) if u != v ])
        }
    
        vertices = graph['vertices']
        for vertice in graph['vertices']:
            self.make_set(vertice)

        minimum_spanning_tree = set()
        forbidden_src = set()
        forbidden_dst = set()
        edges = list(graph['edges'])
        edges.sort()
        for edge in edges:
            weight, vertice1, vertice2 = edge
            if find(vertice1) != find(vertice2): # not cycle
                if (vertice1 not in forbidden_src) and (vertice2 not in forbidden_dst): # path restriction
                    self.union(vertice1, vertice2, vertices)
                    minimum_spanning_tree.add(edge)
                    forbidden_src.add(vertice1)
                    forbidden_dst.add(vertice2)

        self.solution = [self.parent[vertices[0]]]
        self.cost = 0
        # print(minimum_spanning_tree)
        for i in range(N - 1):
            curr = self.solution[i]
            # print(i, curr)
            for w, u, v in minimum_spanning_tree:
                if u == curr:
                    self.solution.append(v)
                    self.cost += w
                    break
        return self