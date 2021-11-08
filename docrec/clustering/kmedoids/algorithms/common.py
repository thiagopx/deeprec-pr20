import numpy as np
from numba import jit


def pick_node(n, k):
    perm = np.random.permutation(n)
    return perm[:k], perm[k:]


def pick_neighbor(node, others):
    n1, n2 = node.size, others.size
    i, j = np.random.randint(n1), np.random.randint(n2)
    return others[j], i, node[i], j


@jit(nopython=True)
def calculate_top2(top2, pwdist):

    # iterate over data points
    for i in range(pwdist.shape[0]):
        m1 = 0
        m2 = 1

        if pwdist[i, m1] > pwdist[i, m2]:
            m1, m2 = m2, m1

        # for each medoid/cluster
        for j in range(2, pwdist.shape[1]):
            if pwdist[i, j] < pwdist[i, m1]:
                m2 = m1
                m1 = j
            elif pwdist[i, j] < pwdist[i, m2]:
                m2 = j

        top2[i, 0] = m1
        top2[i, 1] = m2


@jit(nopython=True)
def swap_cost(pwdist, c, top2, clust):
    total_cost = 0

    for i in range(pwdist.shape[0]):
        ref_c = top2[i, 1] if clust[i] == c else top2[i, 0]
        new_c = ref_c if pwdist[i, ref_c] <= pwdist[i, c] else c
        total_cost += pwdist[i, new_c]

    return total_cost


@jit(nopython=True)
def print_total_cost(mdist_n):
    total_cost = 0
    for i in range(mdist_n.shape[0]):
        m = mdist_n[i, 0]
        for v in mdist_n[i, 1:]:
            if v < m:
                m = v
        total_cost += m
    return total_cost