# PRC Clustering and Classification Using sklearn


To use, you'll need to 

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
    >>> ams = similarity.KNN(10)
    >>> c = PinchRatioClustering(n_clusters=2, 
    ...                          adj_matrix_strategy=ams,
    ...                          n_trials=1)
    >>> c.fit(data)
    >>> adjusted_rand_score(c.labels, labels)
    1.0
    
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
(We only use some of the iris data for speed -- this code is run with
the tests in tests.py)

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


