import numpy as np


def nearest_neighbor(
    matrix, mode='repetitive', inverse=False, start=0, seed=None
):
    
    assert mode in ('random', 'fixed', 'repetitive')
    
    n = matrix.shape[0]
    matrix = np.ma.array(matrix, mask=False)
   
    # Start vertex (random, all, fixed)
    if mode == 'repetitive':
        start = range(n)
    elif mode == 'random':
        if seed is not None:
            np.random.seed(seed)
        start = [np.random.randint(0, n)]
    else:
        start = [start]
   
    # Auxiliar functions
    path = []
    insert = lambda v: path.insert(0, v) if inverse else path.append(v)
    visit = lambda v: (
        matrix.mask[v, :] if inverse else matrix.mask[:, v]
    ).fill(True)
    current = lambda: path[0] if inverse else path[-1]
    best_neighbor = lambda v: (
        matrix[:, v] if inverse else matrix[v, :]
    ).argmin()
    get_cost = lambda v, w: matrix[w, v] if inverse else matrix[v, w]
    
    # Search best solution for each start vextex
    min_cost = float('inf')
    best = None
    for v in start:
        path[:] = [v]
        visit(v)
        cost = 0
        for i in range(n - 1):
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
    
    return best, min_cost
