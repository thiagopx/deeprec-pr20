import numpy as np
from itertools import product

from .common import pick_node
from .common import calculate_top2, swap_cost

def pam(pwdist, k, seed, verbose, max_it):
   # Seeding pseudo-random number generator
   np.random.seed(seed)

   # Number of buildcostmatrix (data points)
   n = pwdist.shape[0]

   # Auxiliar structure which assigns for each data point the 2 closest medoids (clusters)
   top2 = np.empty((pwdist.shape[0],2), dtype=np.int32)

   #================================================================================================
   # Global search process
   #================================================================================================

   # Random node G_nk (medoids set)
   medoids, non_medoids = pick_node(n,k)
   # Distances from data points to medoids
   medoids_dist = pwdist[:,medoids]
   # Calculate for each data point the two best clusters
   calculate_top2(top2, medoids_dist)
   # Clustering result
   clust = top2[:,0].copy()
   # Total cost (initially is taken as the local best cost)
   g_best_cost = np.sum(medoids_dist[np.arange(n), clust])

   i = 1
   while i <= max_it:
      #=============================================================================================
      # Best neighbor search (local search)
      #=============================================================================================
      l_best_cost = float('Inf')
      swap = None

      for (medoids_idx, old_medoid), (non_medoids_idx, new_medoid) \
          in product(enumerate(medoids), enumerate(non_medoids)):
         # Update distances according to the neighbor
         medoids_dist[:,medoids_idx] = pwdist[:,new_medoid]
         # Calculate swap cost
         cost = swap_cost(medoids_dist, medoids_idx, top2, clust)

         # Is the local best?
         if cost < l_best_cost:
            l_best_cost = cost
            swap = (medoids_idx, old_medoid, non_medoids_idx, new_medoid)

         # Restore distance information
         medoids_dist[:,medoids_idx] = pwdist[:,old_medoid]

      # Is the global best?
      if l_best_cost >= g_best_cost:
         break

      g_best_cost = l_best_cost
      medoids_idx, old_medoid, non_medoids_idx, new_medoid = swap
      medoids[medoids_idx], non_medoids[non_medoids_idx] = new_medoid, old_medoid
      medoids_dist[:,medoids_idx] = pwdist[:,new_medoid]
      calculate_top2(top2, medoids_dist)
      clust[:] = top2[:,0]

#       print i, g_best_cost
      i += 1

   clusters = [[medoid] + non_medoids[clust[non_medoids]==c].tolist()
               for c, medoid in enumerate(medoids)]

   return clusters, g_best_cost