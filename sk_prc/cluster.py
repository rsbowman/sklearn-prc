import sys
from Queue import PriorityQueue

import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
import prc

#np.set_printoptions(precision=3, suppress=True)
        
def _compute_ratios(lst, compute_thick,
                    min_thick=0.0001):
    ratios = np.zeros_like(lst)
    ratios[0] = ratios[-1] = 1.0

    for i in range(1, len(ratios) - 1):
        thick_part = compute_thick(i, lst)
        ratios[i] = lst[i] / max(thick_part, min_thick)

    return ratios
    
def pinch_ratios(boundaries):
    def compute_thick(i, b):
        return min(b[:i].max(), b[i + 1:].max())
    return _compute_ratios(boundaries, compute_thick)

def normalized_ratios(boundaries):
    def compute_thick(i, b):
        return min(b[:i].sum(), b[i + 1:].sum())
    return _compute_ratios(boundaries, compute_thick)

def sparse_ratios(boundaries):
    def compute_thick(i, b):
        return min(i + 1, len(b) - i)
    return _compute_ratios(boundaries, compute_thick)

#######################
#
# Clustering Algorithms
#
#######################

def compute_pr_cluster_indices(ordering, boundary, n_clusters,
                               compute_thick_part):
    def find_split(C, boundary):
        best_pr = sys.float_info.max
        index = -1
        for i in range(1, len(C) - 2):
            if (boundary[C[i - 1]] >= boundary[C[i]] and 
                boundary[C[i]] <= boundary[C[i + 1]]):
                thick_width = compute_thick_part(boundary, C, i)
                pr = boundary[C[i]] / thick_width
                if pr < best_pr:
                    best_pr = pr
                    index = i
        return best_pr, index + 1

    pinch_ratios = []
    queue = PriorityQueue()
    C = range(len(ordering))
    pr, idx = find_split(C, boundary)
    queue.put((pr, idx, C))
    pinch_ratios.append(pr)
    while queue.qsize() < n_clusters:
        q, t, C_i = queue.get()
        if t < 0:
            # no split loc
            break
        C_j, C_k = C_i[:t], C_i[t:]
        pr1, idx1 = find_split(C_j, boundary)
        pr2, idx2 = find_split(C_k, boundary)
        pinch_ratios.append(pr1)
        pinch_ratios.append(pr2)
        queue.put((pr1, idx1, C_j))
        queue.put((pr2, idx2, C_k))

    cluster_indices, n_actual_clusters = [], 0
    for i in range(min(queue.qsize(), n_clusters)):
        q, t, C_i = queue.get()
        if C_i:
            cluster_indices.append(C_i)
            n_actual_clusters += 1

    pinch_ratios = sorted(pinch_ratios)[:n_actual_clusters - 1]
    return pinch_ratios, cluster_indices
    
## base class
class CutClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, adj_matrix_strategy,
                 n_trials=1, initial_ordering=None):
        self.n_clusters = n_clusters
        self.adj_matrix_strategy = adj_matrix_strategy
        self.n_trials = n_trials
        self.initial_ordering = initial_ordering
        self._pinch_ratios = None
        
    def fit(self, X):
        from numpy.random.mtrand import RandomState
        min_width = np.repeat(sys.float_info.max, len(X) - 1)
        randomizer = RandomState(111)
        if self.initial_ordering is not None:
            ordering = self.initial_ordering
            assert len(ordering) == len(X), \
                'initial_ordering has wrong size'
        else:
            ordering = np.arange(len(X), dtype=int)
        for i in range(self.n_trials):
            final_ordering, bd, labels, prs = self._fit_once(X, ordering)
            width = np.sort(bd)
            if lt_lex(width, min_width):
                best_order = final_ordering
                best_bd = bd
                best_labels = labels
                best_pinch_ratios = prs
                min_width = width
            randomizer.shuffle(ordering)

        self._ordering = best_order
        self._boundary = best_bd
        self.labels_ = best_labels
        self._pinch_ratios = best_pinch_ratios
        
    @property
    def labels(self):
        return self.labels_
        
    @property
    def pinch_ratios(self):
        return self._pinch_ratios #pinch_ratios(self._boundary)

    @property
    def ordering(self):
        return self._ordering

    @property
    def boundary(self):
        return self._boundary

    @property
    def width(self):
        return np.sort(self._boundary)

def lt_lex(a, b):
    """ return true if array a compares < to array b lexicographically 
    """
    for i in range(len(a)):
        if a[i] < b[i]: return True
        elif a[i] > b[i]: return False
    return False

def compute_thick_part_PR(boundary, C, i):
    return min(boundary[C[:i]].max(),
               boundary[C[i + 1:-1]].max())
    
class PinchRatioCppClustering(CutClustering):
    def __init__(self, n_clusters, adj_matrix_strategy,
                 n_trials=1, initial_ordering=None):
        super(PinchRatioCppClustering, self).__init__(
            n_clusters, adj_matrix_strategy,
            n_trials, initial_ordering)

    def _fit_once(self, X, initial_order):
        adj_matrix = self.adj_matrix_strategy(X)
        order = prc.createOrder(initial_order)
        policy = prc.prcPolicyStruct()
        policy.prcRecurseTILO = True
        policy.prcRefineTILO = True
        labels = prc.ivec([0] * len(X))
        res = prc.pinchRatioClustering(adj_matrix, order,
                                       labels, self.n_clusters, policy)
        ordering = np.fromiter(order.vdata, dtype=int)
        boundary = np.fromiter(order.b.b, dtype=float)[:-1]
        labels = np.fromiter(labels, dtype=int)
        pinch_ratios, _ = compute_pr_cluster_indices(
            ordering, boundary, self.n_clusters,
            compute_thick_part_PR)
        return ordering, boundary, labels, pinch_ratios

class PinchRatioClustering(CutClustering):
    def __init__(self, n_clusters, adj_matrix_strategy,
                 n_trials=1, initial_ordering=None):
        super(PinchRatioClustering, self).__init__(
            n_clusters, adj_matrix_strategy,
            n_trials, initial_ordering)
        
    def _fit_once(self, X, initial_order):
        adj_matrix = self.adj_matrix_strategy(X)
        N = adj_matrix.shape[0]
        degrees = adj_matrix.sum(axis=1)
        boundary = np.zeros(N)

        ordering = prc.createOrder(initial_order)
        policy = prc.tiloPolicyStruct()
        prc.TILO(adj_matrix, ordering, policy)
        
        boundary = np.fromiter(ordering.b.b, dtype=float)[:-1]
        ordering = np.fromiter(ordering.vdata, dtype=int)

        #print 'BDR', boundary
        #print 'PRS', pinch_ratios(boundary)
        #print 'ORD', ordering
        pinch_ratios, clusters = self._find_clusters(ordering, boundary)
        labels = np.zeros(N, dtype=int)
        for i, cluster in enumerate(clusters):
            labels[cluster] = i

        return ordering, boundary, labels, pinch_ratios

    def _compute_thick_part(self, boundary, C, i):
        return compute_thick_part_PR(boundary, C, i)
        
    def _find_clusters(self, ordering, boundary):
        prs, cluster_indices = compute_pr_cluster_indices(
            ordering, boundary, self.n_clusters,
            self._compute_thick_part)

        clusters = []
        for C in cluster_indices:
            clusters.append(np.array([ordering[v] for v in C]))

        return prs, clusters
        
    @property
    def labels(self):
        return self.labels_


class NormalizedCutClustering(PinchRatioClustering):
    def _compute_thick_part(self, boundary, C, i):
        return min(boundary[C[:i]].sum(),
                   boundary[C[i + 1:-1]].sum())

class SparseCutClustering(PinchRatioClustering):
    def _compute_thick_part(self, boundary, C, i):
        return min(i + 1, len(boundary) - i)

############################################
## routines below for standalone/command line clustering

def create_seed_order(data, cluster_algo):
    cluster_algo.fit(data)
    labels = cluster_algo.labels_
    return np.concatenate(
        [np.where(labels == i)[0] for i in np.unique(labels)])

def main(argv):
    import argparse, random
    import scipy
    from sklearn import cluster, metrics, clone
    import similarity
    from ttable import TextTable

    def parse_features(s):
        return [int(i) for i in s.split(',')]
        
    parser = argparse.ArgumentParser(
        description='Output ARI scores for various clustering algorithms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=scipy.loadtxt,
                        help='data in space delimited form')
    parser.add_argument('labels', type=scipy.loadtxt,
                        help="labels in space delimited form")
    parser.add_argument('-k', type=int, default=-1,
                        help="k value for k-nearest neighbors")
    parser.add_argument('--sigma', type=float, default=-1.0,
                        help="sigma value for Gaussian adj. matrix")    
    parser.add_argument('--trials', type=int, default=10,
                        help="number of trials")
    parser.add_argument('--features', type=parse_features, default=[],
                        help="comma sep. list of feature indices to include")
    parser.add_argument('--exclude', type=parse_features, default=[],
                        help="comma sep. list of feature indices to exclude")
    parser.add_argument('--max-samples', type=int, default=-1,
                        help="sample data so it has this size")
    
    args = parser.parse_args()

    n_trials = args.trials
    knn_strat = similarity.KNN(args.k)
    gauss_strat = similarity.Gaussian(args.sigma)

    if args.max_samples > 0:
        indices = np.array(random.sample(xrange(len(args.data)),
                                         args.max_samples))
        data = args.data[indices]
        labels = args.labels[indices]
    else:
        data = args.data
        labels = args.labels

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
        
    if args.features and args.exclude:
        print "can't both include and exclude features"
        return 1
        
    if args.features:
        data = data[:, args.features]
    if args.exclude:
        included_indices = np.delete(np.arange(len(data[0])),
                                     args.exclude)
        data = data[:, included_indices]

    
    # class SpectralWrapper(BaseEstimator):
    #     def __init__(self, n_clusters, adj_strat):
    #         self._c = cluster.SpectralClustering(n_clusters,
    #                                              affinity='precomputed')
    #         self.adj_strat = adj_strat
    #     def fit(self, X):
    #         self._fitted = self._c.fit(X)
    #         self.labels_ = self._fitted.labels_

    gauss_adj_matrix = similarity.Gaussian(args.sigma)(data)
    knn_adj_matrix = similarity.KNN(args.k)(data)
    
    prc_classifiers = [
        ('PRC', PinchRatioClustering(
            n_clusters, similarity.AdjacencyMatrix(), n_trials)),
        ('PRC-C++', PinchRatioCppClustering(
            n_clusters, similarity.AdjacencyMatrix(), n_trials)),
        ('Norm', NormalizedCutClustering(
            n_clusters, similarity.AdjacencyMatrix(), n_trials)),
        ('Sparse', SparseCutClustering(
            n_clusters, similarity.AdjacencyMatrix(), n_trials)),
        ]
    comparison_classifiers = [
        ('Affinity', cluster.AffinityPropagation().fit(data)),
        ('KMeans', cluster.KMeans(n_clusters).fit(data)),
        ('MeanShift', cluster.MeanShift().fit(data)),
        ('DBScan', cluster.DBSCAN().fit(data))
    ]

    try:
        comparison_classifiers.append(
            ('Spectral/Gauss', cluster.SpectralClustering(
                n_clusters, affinity="precomputed").fit(gauss_adj_matrix)))
    except ValueError:
        pass

    try:
        comparison_classifiers.append(
            ('Spectral/KNN', cluster.SpectralClustering(
                n_clusters, affinity="precomputed").fit(knn_adj_matrix)))
    except ValueError:
        pass

    seed_clusterers = [
        cluster.KMeans(n_clusters, random_state=117),
        cluster.AffinityPropagation(),
        cluster.DBSCAN(random_state=117)
        ]
    print 'Using k={}, sigma={:.4}\n'.format(knn_strat.compute_k(data),
                                             gauss_strat.compute_sigma(data))
    prc_table = TextTable(["Method         ", "RAND    ", "KMeans   ",
                           "Affinity  ", "DBSCAN"],
                          ["", " .3f", " .3f", " .3f", " .3f"])
    print prc_table.header()

    table = TextTable(["Method         ", "RAND"],
                      ["", " .3f"])
    for name, classifier in comparison_classifiers:
        guesses = classifier.labels_
        score = metrics.adjusted_rand_score(guesses, labels)
        print table.row(name, score)
    
    for strategy_name, adj_matrix in [("Gauss", gauss_adj_matrix),
                                      ("KNN", knn_adj_matrix)]:
        for name, cl_base in prc_classifiers:
            classifier = clone(cl_base)
            classifier.fit(adj_matrix)
            score = metrics.adjusted_rand_score(
                classifier.labels_, labels)

            initialized_scores = []
            for init_method in seed_clusterers:
                seeded_order = create_seed_order(data, init_method)
                classifier = clone(cl_base)
                classifier.set_params(n_trials=1,
                                      initial_ordering=seeded_order)
                classifier.fit(adj_matrix)
                initialized_scores.append(
                    metrics.adjusted_rand_score(classifier.labels_, labels))

            print prc_table.row(name + "/" + strategy_name,
                                score, *initialized_scores)

        
    return 0
        
if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
