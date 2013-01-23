import numpy as np

# constantly complains about neighbors having same distance:
# def knn_sim_matrix(X, k):
#     n = X.shape[0]
#     if k <= 0:
#         k = int(np.log(n) + 1)
#     G = kneighbors_graph(X, k)
#     m = (0.5 * (G + G.T)).todense()
#     return m

## 
# def knn_sim_matrix(X, k, mode=None):
#     from sklearn.neighbors import BallTree
#     n = X.shape[0]
#     if k <= 0:
#         k = int(np.log(n) + 1)
#     W = np.zeros((n, n), dtype=X.dtype)
#     tree = BallTree(X)
#     for row in range(n):
#         nn_dist, nn_indices = tree.query(X[row], k)
#         for col in nn_indices:
#             W[row, col] = 1
#     return 0.5 * (W + W.T)

def knn_sim_matrix(X, k, mode=None):
    """ X - data as row vectors
    k - number of nearest neighbors, if <=0 then log(n) + 1
    mode - unused; for compatibility with old knn func.
    """
    import scipy.spatial as spatial
    n = X.shape[0]
    if k <= 0:
        k = int(np.log(n)) + 1
    W = np.zeros((n, n), dtype=X.dtype)
    kdt = spatial.cKDTree(X)

    for row in range(n):
        nn_dist, nn_indices = kdt.query(X[row], k)
        for col in nn_indices:
            W[row, col] = 1
        W[row, row] = 0

    m = 0.5 * (W + W.T)
    return m

# def scaled_nbr_matrix(X, k, mode=None):
#     """ X - data as row vectors
#     k - number of nearest neighbors, if <=0 then log(n) + 1
#     mode - unused; for compatibility with old knn func.
#     """
#     import scipy.spatial as spatial
#     n = X.shape[0]
#     if k <= 0:
#         k = int(np.log(n) + 1)
#     W = np.zeros((n, n), dtype=X.dtype)
#     kdt = spatial.cKDTree(X)

#     for row in range(n):
#         nn_dist, nn_indices = kdt.query(X[row], k)
#         for i, col in enumerate(nn_indices):
#             W[row, col] = 1./ 2.0**i
#         W[row, row] = 0.

#     m = 0.5 * (W + W.T)
#     return m
    
def average_knn_dist(X, k):
    from scipy.spatial import cKDTree
    kd_tree = cKDTree(X)
    n = X.shape[0]
    if k <= 0:
        k = int(np.log(n) + 1)
    avg = 0.0
    for i in range(len(X)):
        dists, indices = kd_tree.query(X[i], k)
        avg += dists[k - 1]
    return avg / n
    
def gaussian_sim_matrix(X, sigma=-1):
    r = X.shape[0]
    if sigma <= 0:
        sigma = average_knn_dist(X, -1)
    M = np.dot(X,X.transpose())

    gamma =-1.0/(2.0*sigma*sigma)
    W = np.zeros((r,r),X.dtype)
    for i in range(r):
        W[i,i] = 0.0
        for j in range(i+1,r):
            W[i,j] = np.exp(gamma*(M[i,i]-2*M[i,j]+M[j,j]))
            W[j,i] = W[i,j]
    return W

class SimilarityStrategy(object):
    pass
    
class KNN(SimilarityStrategy):
    def __init__(self, k=-1):
        self._k = k

    def compute_k(self, data):
        if self._k == -1:
            n = data.shape[0]
            return int(np.log(n)) + 1
        else:
            return self._k

    def __call__(self, data):
        return knn_sim_matrix(data, self.compute_k(data))

    def __repr__(self):
        return '<KNN k={}>'.format(self._k)

class Gaussian(SimilarityStrategy):
    def __init__(self, sigma=-1, scale=1.0):
        self._sigma = sigma
        self._scale = scale

    def compute_sigma(self, data):
        if self._sigma == -1:
            sigma = average_knn_dist(data, -1)
        else:
            sigma = self._sigma
        return sigma * self._scale
        
    def __call__(self, data):
        return gaussian_sim_matrix(data, self.compute_sigma(data))

    def __repr__(self):
        return '<Gaussian sigma={:.2f}>'.format(self._sigma)
        
class PeturbedKNN(SimilarityStrategy):
    def __init__(self, k=-1):
        self._k = k
        
    def __call__(self, data):
        scale = 0.001 * data.min() 
        peturbed_data = data + scale * np.random.random(data.shape)
        return knn_sim_matrix(peturbed_data, self._k)

class AdjacencyMatrix(SimilarityStrategy):
    """ strategy which indicates that callers will pass an
    adjacency matrix instead of a list of data points
    """
    def __call__(self, data):
        rows, cols = data.shape
        assert rows == cols, 'adjacency matrix must be square'
        return data

# class ScaledNbrAdjacencyMatrixStrategy(object):
#     def __init__(self, k=-1):
#         self._k = k

#     def __call__(self, data):
#         return scaled_nbr_matrix(data, self._k)
