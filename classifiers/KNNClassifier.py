from base.BaseEstimator import BaseEstimator
import numpy as np

class KNNClassifier(BaseEstimator):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        # yes i know this could have been written much more cleanly
        distances = np.array([np.linalg.norm(x - v) for v in self.X])
        top_k_args = np.argsort(distances)[:self.k]
        d = {}
        for arg in top_k_args:
            v = self.y[arg]
            if d.get(v):
                d[v] += 1
            else:
                d[v] = 1
        highest_key = None
        highest_value = None
        for key, value in d.items():
            if highest_value == None or value > highest_value:
                highest_value = value
                highest_key = key
        return highest_key

    