import numpy as np
from base.BaseEstimator import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(self, n_iter=1000, learning_rate=0.1):
        self.coefficients = None
        self.intercept = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)

        self.coefficients = np.zeros(X.shape[1])
        self.intercept = 0

        for i in range(0, self.n_iter):
            # forward pass, creating the linear combination here
            z = X @ self.coefficients + self.intercept

            # now squashing the z values of the line via a sigmoid function (inverse of logit) to yield values between 0 and 1, which are our probability values 
            p = 1 / (1 + np.exp(-z))   

            # now we compare each probability value to the actual value (0 or 1). logistic regression's error is from this, not R^2 like linear regression
            error = p - y

            # the gradient calculation portion. the error from the last line says how "wrong" each sample is with a sign. 
            # then for each sample we multiply its error by that sample's value for the feature, then averages it out.
            # for the intercept we just see the overall mean error
            dw = X.T @ error / n
            db = np.mean(error)


            # then we use those weights scaled by the learning rates to move the weights
            self.coefficients = self.coefficients - self.learning_rate * dw
            self.intercept = self.intercept - self.learning_rate * db

        return self

    def predict(self, X):
        X = np.asarray(X)

        z = X @ self.coefficients + self.intercept
        p = 1 / (1 + np.exp(-z))

        return np.where(p >= 0.5, 1, 0)

