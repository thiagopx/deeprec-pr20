import libjigsaw
from ..compatibility.andalo import Andalo

def solve_from_strips(strips, d=0):
    ''' Solve RSSTD using Andalo (2017) method from strips. '''

    assert len(strips.strips) > 0
    
    stacked = Andalo(strips).features(d)
    return libjigsaw.solve_from_strips(stacked, len(strips.strips))


def solve_from_matrix(matrix):
    '''
    Solve RSSTD using Andalo (2017) optimization algorithm for a given cost
    matrix.
    '''
    
    solution = libjigsaw.solve_from_matrix(matrix)
    cost = matrix[solution[: -1], solution[1 :]].sum()
    return solution, cost
