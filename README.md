# PRC Clustering and Classification Using sklearn

This is a python package implementing several clustering and
classification algorithms that use [Pinch Ratio Clustering][1].  To
use it, you'll need [scikit-learn][2] (tested with versions 0.12 and
0.13) as well as the python bindings to the [C++ library][3].

Install with

    python setup.py install
    
Run the tests with

    python tests.py
    
to make sure everything works.

## Clustering

The file cluster.py contains sklearn compatible clustering algorithms using
PRC.  An example:

    >>> from sk_prc.cluster import PinchRatioClustering
    >>> from sk_prc import similarity
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.metrics import adjusted_rand_score
    >>> data, labels = make_blobs(100, 2, 2, random_state=106)
    >>> knn_strategy = similarity.KNN(10)
    >>> c = PinchRatioClustering(n_clusters=2, 
    ...                          adj_matrix_strategy=knn_strategy,
    ...                          n_trials=1)
    >>> c.fit(data)
    >>> adjusted_rand_score(c.labels, labels)
    1.0
    
Note that we can set the number of clusters we want, the adjacency
matrix type to use, how many TILO runs to do, and the initial ordering
to use.  Gaussian and k nearest neighbors adjacency matrices are
supported, and if you want to use your own adjacency matrix you can do
that, too.  Right now the TILO run with minimal width (widths sorted
in nondecreasing order!)  over n_trials runs is chosen.

Furthermore, you can get a bunch of information about the clustering,
like the ordering, boundary, pinch ratios, and width:
  
    >>> c.ordering                  # doctest: +ELLIPSIS
    array([15, 35, ..., 59, 86])
    >>> c.boundary[49] == 0.0       # good separation between two blobs
    True
    >>> c.pinch_ratios              # note good separation
    [0.0]
    >>> list(c.width)               # doctest: +ELLIPSIS
    [0.0, 6.0, ..., 50.0, 50.5]

## Classification

**Note** that these classifiers aren't as well tested as the
clustering stuff.  Use at your own risk.

The file classify.py contains BinaryTiloClassifier, which can work
with sklearn to implement classifiers based on TILO/PRC.  It is
parametrized by a cut strategy and an adjacency matrix strategy.

    >>> from sk_prc.classify import BinaryTiloClassifier, NearestCutStrategy
    >>> from sk_prc import similarity
    >>> import numpy as np
    >>> c = BinaryTiloClassifier(NearestCutStrategy(),
    ...                          similarity.Gaussian())
    >>> data = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
    ...                  [15, 15], [14, 14], [14, 15], [15, 14]], dtype=float)
    >>> labels = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    >>> fitted_model = c.fit(data, labels)
    >>> guesses = fitted_model.predict(np.array([[1.5, 1.5],
    ...                                         [11.0, 11.0]]))
    >>> guesses[0], guesses[1]
    ('a', 'b')
    
Here is an example of multiclass classification using bits of sklearn.
(We only use some of the iris data for speed)

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.multiclass import OneVsOneClassifier
    >>> iris = load_iris()
    >>> indices = np.arange(0, 150, 10) ## use a subset of the data for speed
    >>> iris_data, iris_labels = iris.data[indices], iris.target[indices]
    >>> c = BinaryTiloClassifier(NearestCutStrategy(),
    ...                          similarity.KNN(6))
    >>> mcc = OneVsOneClassifier(c)
    >>> guessed_labels = mcc.fit(iris_data, iris_labels).predict(iris_data)
    >>> (guessed_labels != iris_labels).sum()
    1


[1]: http://arxiv.org/abs/1206.0771
[2]: http://scikit-learn.org/
[3]: http://cs.okstate.edu/~doug/src/prc/
