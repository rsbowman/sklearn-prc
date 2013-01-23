from unittest import TestCase, TestSuite, main, makeSuite
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import metrics
import prc

from classify import BinaryTiloClassifier, \
    PinchRatioCutStrategy, NearestCutStrategy, find_closest_min
from cluster import pinch_ratios, normalized_ratios, sparse_ratios
from cluster import PinchRatioClustering, PinchRatioCppClustering
import similarity

np.set_printoptions(precision=2)

class CommonTests(TestCase):
    def test_ratios(self):
        bds = np.array([1,2,3,2,4,3,2,1], dtype=float)
        assert_array_equal(pinch_ratios(bds),
                           np.array([1., 2., 3./2., 2./3.,
                                     4./3., 3./2., 2., 1.]))
        assert_array_equal(normalized_ratios(bds),
                           np.array([1., 2., 1., 1./3.,
                                     2./3., 1., 2., 1.]))
        assert_array_equal(sparse_ratios(bds),
                           np.array([1., 1., 1., .5, 1., 1., 1., 1.]))

class ClusteringTests(TestCase):
    def make_blobs(self, n_samples):
        from sklearn.datasets import make_blobs        
        centers = np.array([[0., 0., 0.], [10., 10., 10.]])
        return make_blobs(n_samples=n_samples, n_features=3,
                          centers=centers, random_state=7)

    def test_n_clusters(self):
        c = PinchRatioClustering(2, similarity.KNN(10))
        data, labels = self.make_blobs(50)
        c.fit(data)
        score = metrics.adjusted_rand_score(c.labels, labels)
        self.assertEqual(score, 1.0)

    def test_n_trials(self):
        c = PinchRatioClustering(2, similarity.KNN(9),
                                 n_trials=5)
        data, labels = self.make_blobs(150)
        c.fit(data)
        score = metrics.adjusted_rand_score(c.labels, labels)
        self.assertEqual(score, 1.0)

    def test_use_adjacency_matrix(self):
        data, labels = self.make_blobs(50)
        adj_matrix = similarity.KNN(10)(data)
        c = PinchRatioClustering(2, similarity.AdjacencyMatrix())
        guessed_labels = c.fit_predict(adj_matrix)
        assert_array_equal(labels, guessed_labels)
        
class BinaryClassifierTests(TestCase):
    def setUp(self):
        self.two_class_pts = np.array(
            [[0.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
             [50, 49, 49], [49, 50, 50], [50, 50, 51], [50, 49, 51]])
        self.two_class_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        self.three_class_pts = np.array(
            [[0.0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
             [50, 0, 0], [50, 1, 0], [50, 0, 1], [50, 1, 1],
             [0, 0, 40], [0, 1, 40], [1, 1, 40], [1, 0, 40]], dtype="d")
        self.three_class_labels = np.array([0, 0, 0, 0,
                                            1, 1, 1, 1,
                                            2, 2, 2, 2])

    def test_predict_return_numpy_array(self):
        b = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.KNN())
        b.fit(np.array([[0, 0, 0], [.1, 0, 0],
                        [.9, 1, 1], [1, 1, 1]]),
              np.array([0, 0, 1, 1]))
        result = b.predict(np.array([[.1, .1, .1], [.9, .9, .9]]))
        self.assertTrue(isinstance(result, np.ndarray))
        
    def test_two_clusters(self):
        pts = self.two_class_pts
        labels = self.two_class_labels
        #c = BinaryTiloClassifier(similarity.KNN(3))
        test_pts = np.array([[0.5, 0.5, 0.5],
                             [55.0, 55.0, 55.0]])
        for adj_strat in (similarity.Gaussian(),
                          similarity.Gaussian(10.0)):
            c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                     adj_strat)
            fitted = c.fit(pts, labels)
            assert_array_equal(fitted.predict(test_pts),
                               np.array([0,1]))
            assert_array_equal(fitted.predict_proba(test_pts),
                               np.array([[1., 0], [0, 1]]))

    def test_cpp_prc_labels(self):
        data = self.three_class_pts
        peturbed_data = data + 0.1 * np.random.random(data.shape)
        c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.Gaussian())
        c.fit(peturbed_data, np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert_array_equal(c.predict(np.array([[0, 0, 0]])),
                           np.array([0]))

    def test_unsorted_labels(self):
        data = self.two_class_pts
        labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.Gaussian())
        c.fit(data, labels)
        assert_array_equal(c.predict(data), labels)
        
    def test_multicluster(self):
        c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.Gaussian())
        ##c = BinaryTiloClassifier(similarity.KNN())
        ##mcc = OneVsRestClassifier(c)
        mcc = OneVsOneClassifier(c)
        data = self.three_class_pts
        classes = self.three_class_labels

        peturbed_data = data + 0.01 * np.random.random(data.shape)
        fitted = mcc.fit(peturbed_data, classes)
        guesses = fitted.predict(peturbed_data)
        assert_array_equal(guesses, classes)

    def test_score_prc(self):
        c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.Gaussian())
        fitted = c.fit(self.two_class_pts, self.two_class_labels)
        assert_array_equal(fitted.model_labels, self.two_class_labels)

    def test_score_prc_different_labels(self):
        c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.Gaussian())
        labels = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        fitted = c.fit(self.two_class_pts, labels)
        assert_array_equal(fitted.model_labels, labels)

        labels2 = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        fitted2 = c.fit(self.two_class_pts, labels2)
        assert_array_equal(fitted.model_labels, labels2)
        
    def test_nearest_cut(self):
        c = BinaryTiloClassifier(NearestCutStrategy(),
                                 similarity.Gaussian())
        fitted = c.fit(self.two_class_pts, self.two_class_labels)
        assert_array_equal(fitted.model_labels, self.two_class_labels)

    def test_string_labels(self):
        c = BinaryTiloClassifier(PinchRatioCutStrategy(),
                                 similarity.Gaussian())
        labels =  np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
        fitted = c.fit(self.two_class_pts, np.array(['a', 'a', 'a', 'a',
                                                        'b', 'b', 'b', 'b']))
        assert_array_equal(fitted.predict(self.two_class_pts),
                           labels)

    def test_two_cluster_readme(self):
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                         [15, 15], [14, 14], [14, 15], [15, 14]],
                        dtype=float)
        labels = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
        c = BinaryTiloClassifier(NearestCutStrategy(),
                                 similarity.Gaussian())
        fitted_model = c.fit(data, labels)
        guesses = fitted_model.predict(np.array([[1.5, 1.5],
                                                 [11.0, 11.0]]))
        assert_array_equal(guesses, np.unique(labels))

    def test_find_closest_min(self):
        arr = [1, 2, 2, 2, 0, 0, 1, 1, 1, 0, 1, 2, 3, 3, 2]
        ##     0     2     4  5  6  7  8  9  10 11 12 13 14
        
        self.assertEqual(find_closest_min(arr, 2), 4)
        self.assertEqual(find_closest_min(arr, 4), 4)
        self.assertEqual(find_closest_min(arr, 5), 5)
        self.assertEqual(find_closest_min(arr, 6), 5)
        self.assertEqual(find_closest_min(arr, 7), 5) # ambiguous
        self.assertEqual(find_closest_min(arr, 8), 9)
        self.assertEqual(find_closest_min(arr, 9), 9)        
        self.assertEqual(find_closest_min(arr, 10), 9)
        self.assertEqual(find_closest_min(arr, 11), 9)
        self.assertEqual(find_closest_min(arr, 12), 12)
        self.assertEqual(find_closest_min(arr, 13), 13)

    def test_move_first_point(self):
        pts = np.array([[0.,0], [0,1], [1,0], [1,1], [0, 0.5], [0.5, 0],
                        [10, 10], [11, 10], [10, 11], [11, 11],
                        [10.5, 11], [10, 10.5]])
        indices = np.arange(len(pts))
        indices[0], indices[10] = 10, 0
        data = pts[indices, :]
        
        knn_strat = similarity.KNN(3)
        gauss_strat = similarity.Gaussian()
        prc = PinchRatioClustering(2, gauss_strat)

        prc.fit(data)
        assert_array_equal(prc.labels, np.array([1, 0, 0, 0, 0, 0,
                                                 1, 1, 1, 1, 0, 1]))

    
def suite():
    from doctest import DocFileSuite
    s = TestSuite()
    #s.addTest(DocFileSuite('README.md'))
    #print 'ADD BACK DOCTESTS!!!!'
    for c in (CommonTests, ClusteringTests, BinaryClassifierTests):
        s.addTest(makeSuite(c))
    return s

if __name__ == '__main__':
    main(defaultTest='suite')
