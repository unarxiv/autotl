import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import cholesky, cho_solve, solve_triangular
from autotl.distance import edit_distance_matrix
from autotl.distance import bourgain_embedding_matrix

class IncrementalGaussianProcess(object):
    def __init__(self, kernel_lambda):
        self.alpha = 1e-10
        self.alpha = 1e-10
        self._k_matrix = None
        self._distance_matrix = None
        self._x = None
        self._y = None
        self._first_fitted = False
        self._l_matrix = None
        self._alpha_vector = None
        self.edit_distance_matrix = edit_distance_matrix
        self.kernel_lambda = kernel_lambda
    @property
    def kernel_matrix(self):
        return self._distance_matrix
    
    def fit(self, train_x, train_y):
        if self._first_fitted:
            self.incremental_fit(train_x, train_y)
        else:
            self.first_fit(train_x, train_y)
    
    @property
    def first_fitted(self):
        return self._first_fitted
    
    def first_fit(self, train_x, train_y):
        train_x, train_y = np.array(train_x), np.array(train_y)
        self._x = np.copy(train_x)
        self._y = np.copy(train_y)
        self._distance_matrix = self.edit_distance_matrix(self.kernel_lambda, self._x)
        self._distance_matrix = bourgain_embedding_matrix(self._distance_matrix)
        self._k_matrix = 1.0 / np.exp(self._distance_matrix)
        self._k_matrix[np.diag_indices_from(self._k_matrix)] += self.alpha
        self._l_matrix

    def incremental_fit(self, train_x, train_y):
        if not self._first_fitted:
            raise ValueError("The first_fit function needs to be called first")
        
        train_x, train_y = np.array(train_x), np.array(train_y)
        

class BayesianOptimizer(object):
    def __init__(self, searcher, t_min, metric, kernel_lambda, beta):
        self.searcher = searcher
        self.t_min = t_min
        self.metric = metric
        self.kernel_lambda = kernel_lambda
        self.beta = beta
