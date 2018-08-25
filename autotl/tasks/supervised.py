from abc import ABC, abstractmethod

class Supervised(ABC):
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    @abstractmethod
    def fit(self, x_train, y_train, time_limit=None):
        pass
    
    @abstractmethod
    def predict(self, x_test):
        pass
    
    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass