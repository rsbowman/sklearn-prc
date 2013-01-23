import sys
import numpy as np

import prc

from sklearn.base import BaseEstimator
from cluster import pinch_ratios

class CutStrategy(object):
    @property
    def pinch_ratios(self):
        b = self._boundary
        return pinch_ratios(self._boundary)

    def boundary(self):
        return self._boundary.copy()
        
    def __call__(self, adj_matrix, n_data1, n_data2):
        N = n_data1 + n_data2
        assert adj_matrix.shape[0] == N

        ordering = prc.createOrder(np.arange(N, dtype="i"))
        policy = prc.tiloPolicyStruct()
        prc.TILO(adj_matrix, ordering, policy)
        boundary = np.fromiter(ordering.b.b, dtype=float)[:-1]
        self._boundary = boundary
        ordering = np.fromiter(ordering.vdata, dtype=int)

        #prc.TILO(adj_matrix, ordering, degrees, boundary)
        #self._boundary = boundary
        
        ## add one because cut is in terms of boundary array
        self._cut = self._find_cut(boundary, n_data1, n_data2) + 1
        
        pred_labels = np.zeros(N, dtype=int)
        pred_labels[ordering[self._cut:]] = 1

        return pred_labels
        
# class PinchRatioCutStrategy(CutStrategy):
#     def __call__(self, adj_matrix, n_data1, n_data2):
#         n = adj_matrix.shape[0]
#         ordering = np.arange(n, dtype="i")
#         prc_labels = np.zeros(n, dtype="i")
#         prc.pinchRatioClustering(adj_matrix, ordering, prc_labels, 2)

#         ## associate integer labels with prc labels: data1 is at beginning
#         ## of data, so is that part of prc_labels mostly 0 or 1?
#         n_ones = prc_labels[:n_data1].sum()
#         if n_ones > int(n_data1 / 2.0): ## reverse zeros and ones
#             pred_labels = 1 - prc_labels
#         else:
#             pred_labels = prc_labels

#         return pred_labels

def find_closest_min(array, index):
    N = len(array)
    rt_counter, lt_counter = index, index
    ## find minimum to right: minimums look like: \__
    while (rt_counter < N - 1 and not
           (array[rt_counter - 1] > array[rt_counter] and
            array[rt_counter + 1] >= array[rt_counter])):
        rt_counter += 1
    ## find minimum to left: minimums look like __/
    while (lt_counter > 0 and not
           (array[lt_counter - 1] >= array[lt_counter] and
            array[lt_counter + 1] > array[lt_counter])):
        lt_counter -= 1
    if lt_counter == 0 and rt_counter == N - 1:
        return index

    dist_left, dist_right = abs(index - lt_counter), abs(index - rt_counter)
    if dist_left > dist_right:
        return rt_counter if rt_counter < N - 1 else index
    elif dist_left < dist_right:
        return lt_counter if lt_counter > 0 else index
    else: ## equal distance
        if lt_counter == 0:
            return rt_counter
        elif rt_counter == N - 1:
            return lt_counter
        elif array[lt_counter] <= array[rt_counter]:
            return lt_counter
        else:
            return rt_counter

class PinchRatioCutStrategy(CutStrategy):
    def _find_cut(self, boundary, n_data1, n_data2):
        return np.argmin(pinch_ratios(boundary))

class NearestCutStrategy(CutStrategy):
    def _find_cut(self, boundary, n_data1, n_data2):
        return find_closest_min(boundary, n_data1 - 1) 

class NormalizedCutStrategy(CutStrategy):
    def _find_cut(self, boundary, n_data1, n_data2):
        return np.argmin(normalized_ratios(boundary))

class SparseCutStrategy(CutStrategy):
    def _find_cut(self, boundary, n_data1, n_data2):
        return np.argmin(sparse_ratios(boundary))
        
# class FixedCutStrategy(CutStrategy):
#     """ Fixed cut strategy always makes the cut between
#     data1 and data2; therefore should not be used for classification
#     """
#     def _find_cut(self, boundary, n_data1, n_data2):
#         return n_data1 - 1
        
class BinaryTiloClassifier(BaseEstimator):
    def __init__(self, cut_strategy, adj_matrix_strategy):
        self.cut_strategy = cut_strategy
        self.adj_matrix_strategy = adj_matrix_strategy
        self._dirty = True ## need to recompute pinch_ratios, etc.
        
    def fit(self, data, labels):
        self._dirty = True
        self._data = data
        self._n = len(data)

        ## get unique labels in the order they appear in the label set
        seen_labels, unique_labels = set(), []
        for l in labels:
            if not l in seen_labels:
                unique_labels.append(l)
                seen_labels.add(l)
        self._unique_labels = np.array(unique_labels)
        
        self._data1 = data[labels == unique_labels[0]]
        self._data2 = data[labels == unique_labels[1]]
        self._labels = labels
        return self

    def _predict_one(self, point):
        data = np.vstack((self._data1, point, self._data2))
        n1 = len(self._data1)
        adj_matrix = self.adj_matrix_strategy(data)
        pred_labels = self.cut_strategy(adj_matrix, n1 + 1, len(self._data2))
        return pred_labels[n1]
        
    def predict(self, data):
        predicted_labels = []
        for point in data:
            predicted_labels.append(
                self._unique_labels[self._predict_one(point)])
        return np.array(predicted_labels)

    def predict_proba(self, data):
        probs = np.zeros((len(data), 2), dtype="d")
        for row, point in enumerate(data):
            probs[row, self._predict_one(point)] = 1.0
        return probs

    def _update_cuts(self):
        if self._dirty:
            data = np.vstack((self._data1, self._data2))
            adj_matrix = self.adj_matrix_strategy(data)
            pl = self.cut_strategy(adj_matrix,
                                   len(self._data1),
                                   len(self._data2))
            self._predicted_labels = pl
            self._pinch_ratios = self.cut_strategy.pinch_ratios
            self._boundary = self.cut_strategy.boundary()
            self._dirty = False

    @property
    def model_labels(self):
        self._update_cuts()
        pred_labels = self._predicted_labels
        new_labels = np.zeros_like(pred_labels)
        new_labels[pred_labels == 0] = self._unique_labels[0]
        new_labels[pred_labels == 1] = self._unique_labels[1]

        return new_labels

    @property
    def pinch_ratios(self):
        self._update_cuts()
        return self._pinch_ratios

def main(argv):
    import scipy
    from sklearn import metrics
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import cross_val_score
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import preprocessing
    import similarity
    
    class ScaledSVC(SVC):
        def _scale(self, data):
            return preprocessing.scale(data)
        def fit(self, X, Y):
            return super(ScaledSVC, self).fit(self._scale(X), Y)
        def predict(self, X):
            return super(ScaledSVC, self).predict(self._scale(X))

    data, labels = scipy.loadtxt(argv[1]), scipy.loadtxt(argv[2])
    if len(argv) > 3:
        features = np.array([int(s) for s in argv[3].split(',')])
        data = data[:, features]
        
    def ovo(model, adj_strat):
        return OneVsOneClassifier(BinaryTiloClassifier(model, adj_strat))

    classifiers = [
        ('TILO/PRC/Gaussian',
         ovo(PinchRatioCutStrategy(),
             similarity.Gaussian())),
        ("TILO/Nearest/Gaussian",
         ovo(NearestCutStrategy(),
             similarity.Gaussian())),
        ("TILO/PRC/KNN",
         ovo(PinchRatioCutStrategy(),
             similarity.KNN())),
        ("TILO/Nearest/KNN",
         ovo(NearestCutStrategy(),
             similarity.KNN())),
        ("SVC", ScaledSVC()),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("K Neighbors", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier())]
    format_str = '{:<30} {} {} {}'
    print '{:<30} {:<10}         RAND   Accuracy'.format('method', 'accuracy')
    for name, c in classifiers:
        scores = cross_val_score(c, data, labels, cv=5)
        #scores = np.array([1., 1.])
        model = c.fit(data, labels)
        guesses = model.predict(data)
        acc = metrics.zero_one_score(guesses, labels)
        rand = metrics.adjusted_rand_score(guesses, labels)
        print '{:<30} {:.4f} +/- {:.4f} {: .4f} {:.4f}'.format(name, scores.mean(),
                                                               scores.std() / 2,
                                                               rand, acc)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
