from base.BaseEstimator import BaseEstimator
import numpy as np

class LinearRegression(BaseEstimator):
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.asarray(X)

        # we add a column of 1s at the front of X to find the intercept
        X = np.column_stack((np.ones(len(X)), X))


        # this is just the closed-form version of the linear regression formula after all the calculus
        # i cannot explain the calculus, i'm just following the linear regression formula here
        square = X.T @ X
        vector = X.T @ y
        w = np.linalg.solve(square, vector)

        # at the end, we have w which is a vector of [intercept, weight1, weight2, ...]
        self.intercept = w[0]
        self.coefficients = w[1:]

        return self

    def predict(self, X):
        predictions = np.asarray(X) @ self.coefficients + self.intercept
        return predictions