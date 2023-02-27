import numpy as np
from sklearn.mixture import GaussianMixture

def gaus_mixture(behaviorMatrix):

  """Performs gaussian mixture model clustering.
  Args:
    data: an n-by-1 numpy array of numbers with n data points
    n_components: a list of digits that are possible candidates for the number of clusters to use
  Returns:
    A single digit (which is an element from n_components) that results in the lowest
    BIC when it is used as the number of clusters to fit a GMM
  """
  # initialize best number of clusters to first element in n_components by
  # (1) fitting a GMM on `data` using the first element in `n_components` as the number
  # of clusters (remember to set random_state=0 when you call GaussianMixture()),
  # (2) calculating the bic on `data` and making it the best bic, and (3) setting the
  # corresponding number of cluster (i.e., the first element of `n_components`
  # as the best number of clusters
  n_components = [1,2,3,4,5,6,7,8,9,10]
  
  gm = GaussianMixture(n_components[0], max_iter=1000, tol=1e-4, init_params='kmeans')
  best_bic = gm.fit(behaviorMatrix).bic(behaviorMatrix)
  best_no_clusters = n_components[0]
  
  for k in n_components:
      gm = GaussianMixture(k, max_iter=1000, tol=1e-4, init_params='kmeans')
      bic = gm.fit(behaviorMatrix).bic(behaviorMatrix)
      if bic < best_bic:
          best_bic = bic
          best_no_clusters = k
  
  return best_no_clusters

