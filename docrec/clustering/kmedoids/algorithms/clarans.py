import numpy as np

from docrec.clustering.kmeansplusplus import kmeans_plus_plus
from .common import pick_node, pick_neighbor
from .common import calculate_top2, swap_cost

def clarans(
    pwdist, k, init='random', seed=None, verbose=False, num_local=2,
    max_neighbor=200
):
    '''
    References:

    CLARANS: A Method for Clustering Objects for Spatial Data Mining

    Algorithm sketch

       1. Input parameters numlocal and maxneighbor. Initialize i to 1, and mincost to a large number.
       2. Set current to an arbitrary node in G_{n,k}.
       3. Set j to 1.
       4. Consider a random neighbor S of current, and based on Equation (5) calculate the cost
          differential of the two nodes.
       5. If S has a lower cost, set current to S, and go to Step (3).
       6. Otherwise, increment j by 1. If j <= maxneighbor, go to Step (4).
       7. Otherwise, when j > maxneighbor, compare the cost of current with mincost. If the former is
          less than mincost, set mincost to the cost of current, and set bestnode to current.
       8. Increment i by 1. If i > numlocal, output bestnode and halt. Otherwise, go to Step (2).
   '''

    assert init in ['random', 'k_means++']

    # Seeding pseudo-random number generator
    np.random.seed(seed)
    # Number of exemplars (data points)
    n = pwdist.shape[0]
    # Auxiliar structure which assigns for each data point the 2 closest
    # medoids (clusters)
    top2 = np.empty((n, 2), dtype=np.int32)

    #=========================================================================
    # Global search process
    #=========================================================================
    best_cost = float('Inf')
    # Final clustering
    clust = np.empty((n,), dtype=np.int32)
    # Final medoids set
    medoids = np.empty((k,), dtype=np.int32)
    # Final non-medoids set
    non_medoids = np.empty((n - k,), dtype=np.int32)

    for i in range(num_local):
        # Random node G_nk (medoids set)
        current, others = pick_node(n, k) if init == 'random' else \
            kmeans_plus_plus(pwdist, k)
        # Distances from data points to medoids
        current_dist = pwdist[:, current]
        # Calculate for each data point the two best clusters
        calculate_top2(top2, current_dist)
        # Clustering
        current_clust = top2[:, 0].copy()
        # Total cost (initially is taken as the local best cost)
        current_cost = np.sum(current_dist[np.arange(n), current_clust])

        # Best neighbor search (local search)
        j = 1
        while j <= max_neighbor:
            # Pick (randomly) a neighbor.
            new_medoid, cluster_idx, old_medoid, others_idx = pick_neighbor(
                current, others
            )
            # Update distances according to the neighbor
            current_dist[:, cluster_idx] = pwdist[:, new_medoid]
            # Calculate swap cost
            cost = swap_cost(current_dist, cluster_idx, top2, current_clust)

            # Is the local best?
            if cost < current_cost:
                current_cost = cost
                current[cluster_idx] = new_medoid
                others[others_idx] = old_medoid
                calculate_top2(top2, current_dist)
                current_clust[:] = top2[:, 0]
                # Reset search counter for the neighbor
                j = 1
            else:
                # Restore distance information
                current_dist[:, cluster_idx] = pwdist[:, old_medoid]
                j += 1

            if verbose:
                print('Local search {} (neigh. {}): cost={:.3f} lbest={:.3f} gbest={.:3f}'.format(
                   i, j, cost, current_cost, best_cost
                ))


        # Is the global best?
        if current_cost < best_cost:
            best_cost = current_cost
            medoids[:] = current
            non_medoids[:] = others
            clust[:] = current_clust
            # print best_cost

        if verbose:
            print('Local search {}: lbest={:.3f} gbest={:.3f}'.format(i, current_cost, best_cost))

    clusters = [[medoid] + non_medoids[clust[non_medoids] == c].tolist()
                for c, medoid in enumerate(medoids)]
    return clusters, best_cost
