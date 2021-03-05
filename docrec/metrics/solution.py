import numpy as np


def neighbor_comparison(solution, init_perm, sizes):
    ''' Accuracy by neighbor comparison. '''

    assert len(solution) > 0
    assert len(init_perm) > 0
    assert len(sizes) > 0

    N = len(solution)
    solution = [init_perm[s] for s in solution]
    num_correct = 0
    neighbors = {}
    id_ = 0
    for size in sizes:
        for _ in range(size - 1):
            neighbors[id_] = [id_ + 1]
            id_ += 1
        first = id_ - size + 1
        first_ = 0
        neighbors[id_] = []
        for size in sizes:
            if first_ != first:
                neighbors[id_].append(first_)
            first_ += size
        id_ += 1
    for i in range(N - 1):
        if solution[i + 1] in neighbors[solution[i]]:
            num_correct += 1
    return num_correct / (N - 1)
