import numpy as np


def kmeans_plus_plus(X, k, n_local_trials=None):
    '''
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence.

    [1] Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful
    seeding". ACM-SIAM symposium on Discrete algorithms. 2007

    Adapted from:
    https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/cluster/k_means_.py#L704
    '''

    n = X.shape[0]

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(k))

    # Pick first center randomly
    center_id = np.random.randint(n)
    distances = X[center_id, :]
    current_potential = distances.sum()
    centers = [center_id]
    
    # Pick the remaining elements
    for c in xrange(1, k):

        # Choose candidates by sampling with probability proportional
        # to the distance value of the closest existing center
        rand_values = np.random.random(n_local_trials) * current_potential
        candidate_ids = np.searchsorted(distances.cumsum(), rand_values)

        # Compute distances to center candidates
        distance_to_candidates = X[candidate_ids, :]

        # Decide which candidate is the best
        best_candidate = None
        best_potential = None
        best_distances = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_distances = np.minimum(
                distances, distance_to_candidates[trial, :]
            )
            new_potential = new_distances.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_potential < best_potential):
                best_candidate = candidate_ids[trial]
                best_potential = new_potential
                best_distances = new_distances

        # Permanently add best center candidate found in local tries
        centers.append(best_candidate)
        current_potential = best_potential
        distances = best_distances
        
        if current_potential == 0:
            break
        
    # Complete k centers
    others = list(set(range(n)) - set(centers))
    np.random.shuffle(others)
    centers += others[: k - c - 1]
    others = others[k - c - 1 :]
    return np.array(centers, dtype=np.int32), np.array(others, dtype=np.int32)
