import numpy as np
from scipy import stats
# from docrec.solver.solverkbh import kruskal_based
# from .solution import accuracy

# def perfect_matchings(matrix, pre_process=False, normalized=False):

#     M = np.array(matrix)
#     assert M.ndim in (2, 3)

#     if pre_process:
#         np.fill_diagonal(M, 0)
#         M = M.max() - M
#         np.fill_diagonal(M, 1e7)

#     num_correct = 0
#     N = M.shape[1]
#     for i in range(N - 1):
#         row_min = M[i].min()
#         col_min = M[:, i + 1].min()
#         if M[i, i + 1] == row_min and M[i, i + 1] == col_min: # minimum
#             if np.sum(M[i] == row_min) == 1 and  np.sum(M[:, i + 1] == col_min) == 1: # uniqueness
#                 num_correct += 1

#     i, j = N - 1, 0
#     row_max = M[i, : -1].max() # exclude diagonal
#     col_max = M[1 :, j].max()  # exclude diagonal
#     if M[i, j] == row_max and M[i, j] == col_max: # minimum
#         num_correct += 1

#     if normalized:
#         return num_correct / N
#     return num_correct


def perfect_matchings(matrix, pre_process=False, normalized=False):

    M = np.array(matrix)
    assert M.ndim in (2, 3)

    if pre_process:
        np.fill_diagonal(M, 0)
        M = M.max() - M
        np.fill_diagonal(M, 1e7)

    num_correct = 0
    N = M.shape[1]
    for i in range(N - 1):
        row_min = M[i].min()
        col_min = M[:, i + 1].min()
        if M[i, i + 1] == row_min and M[i, i + 1] == col_min: # minimum
            if np.sum(M[i] == row_min) == 1 and  np.sum(M[:, i + 1] == col_min) == 1: # uniqueness
                num_correct += 1

    if normalized:
        return num_correct / (N - 1)
    return num_correct


# def precision_mc(matrix, pre_process=False, normalized=False):
#     ''' Kruskal-based algorithm. '''

#     M = np.array(matrix)
#     assert M.ndim in (2, 3)

#     if pre_process:
#         np.fill_diagonal(M, 0)
#         M = M.max() - M
#         np.fill_diagonal(M, 1e7)

#     num_correct = 0
#     N = M.shape[1]
#     for _ in range(N - 1):
#         y, x = np.where(M == M.min())
#         i, j = y[0], x[0]
#         if i + 1 == j and y.size == 1:
#             num_correct += 1
#         M[i, :] = 1e7
#         M[:, j] = 1e7
#     if normalized:
#         #return (num_correct + 1) / N
#         return num_correct / (N - 1)
#     return num_correct


# def precision_mc(matrix, pre_process=False, normalized=False):
#     ''' Kruskal-based algorithm. '''

#     M = np.array(matrix)
#     assert M.ndim in (2, 3)

#     if pre_process:
#         np.fill_diagonal(M, 0)
#         M = M.max() - M
#         np.fill_diagonal(M, 1e7)

#     N = M.shape[1]
#     graph = {
#         'vertices': [v for v in range(N)],
#         'edges': set([(M[u, v], u, v) for u in range(N) for v in range(N) if u != v ])
#     }
#     solution = kruskal_based(graph)
#     # print(solution)
#     return accuracy(solution)


# def precision_nn(matrix, pre_process=False, normalized=False):

#     M = np.array(matrix)
#     assert M.ndim in (2, 3)

#     if pre_process:
#         np.fill_diagonal(M, 0)
#         M = M.max() - M
#         np.fill_diagonal(M, 1e7)

#     N = M.shape[1]

#     y, x = np.where(M == M.min())
#     num_correct = int(y[0] + 1 == x[0])
#     start = y[0]
#     current = x[0]
#     M[:, start] = 1.e7
#     for _ in range(N - 1):
#         M[:, current] = 1e7
#         costs = M[current, :]
#         x = np.where(costs == costs.min())[0]
#         if (current + 1) in x:
#             num_correct += 1
#             next_ = current + 1
#         else:
#             next_ = x[0]
#         current = next_

#     if normalized:
#         return num_correct / (N - 1)
#     return num_correct


# def precision_nn(matrix, pre_process=False):

#     M = np.array(matrix)
#     assert M.ndim in (2, 3)

#     if pre_process:
#         M = M.max() - M
#         np.fill_diagonal(M, 1e7)
#     best = 0
#     N = M.shape[1]
#     M_ = np.empty_like(M)
#     for start in range(N):
#         M_[:] = M
#         num_correct = 0
#         current = start
#         for _ in range(N - 1):
#             M_[:, current] = 1e7
#             costs = M_[current, :]
#             x = np.where(costs == costs.min())[0]
#             if (current + 1) in x:
#                 num_correct += 1
#                 next_ = current + 1
#             else:
#                 next_ = x[0]
#             current = next_
#         if num_correct > best:
#             best = num_correct
#     return best / (N - 1)