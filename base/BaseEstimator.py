from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
