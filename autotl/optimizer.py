import time
from copy import deepcopy
from functools import total_ordering
from queue import PriorityQueue

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import linear_sum_assignment

from autotl.distance import (bourgain_embedding_matrix, edit_distance,
                             edit_distance_matrix)


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
        self._distance_matrix = self.edit_distance_matrix(
            self.kernel_lambda, self._x)
        self._distance_matrix = bourgain_embedding_matrix(
            self._distance_matrix)
        self._k_matrix = 1.0 / np.exp(self._distance_matrix)
        self._k_matrix[np.diag_indices_from(self._k_matrix)] += self.alpha
        self._l_matrix = cholesky(self._k_matrix, lower=True)
        self._alpha_vector = cho_solve((self._l_matrix, True), self._y)
        self._first_fitted = True
        return self

    def incremental_fit(self, train_x, train_y):
        if not self._first_fitted:
            raise ValueError("The first_fit function needs to be called first")

        train_x, train_y = np.array(train_x), np.array(train_y)
        up_right_k = self.edit_distance_matrix(self.kernel_lambda, self._x,
                                               train_x)
        down_left_k = np.transpose(up_right_k)
        down_right_k = self.edit_distance_matrix(self.kernel_lambda, train_x)
        up_k = np.concatenate((self._distance_matrix, up_right_k), axis=1)
        down_k = np.concatenate((down_left_k, down_right_k), axis=1)
        self._distance_matrix = np.concatenate((up_k, down_k), axis=0)
        self._distance_matrix = bourgain_embedding_matrix(
            self._distance_matrix)
        self._k_matrix = 1.0 / np.exp(self._distance_matrix)
        diagonal = np.diag_indices_from(self._k_matrix)
        diagonal = (diagonal[0][-len(train_x):], diagonal[1][-len(train_x):])
        self._k_matrix[diagonal] += self.alpha
        self._x = np.concatenate((self._x, train_x), axis=0)
        self._y = np.concatenate((self._y, train_y), axis=0)

        self._l_matrix = cholesky(self._k_matrix, lower=True)  # Line 2

        self._alpha_vector = cho_solve((self._l_matrix, True),
                                       self._y)  # Line 3

        return self


class BayesianOptimizer(object):
    def __init__(self, searcher, t_min, metric, kernel_lambda, beta):
        self.searcher = searcher
        self.t_min = t_min
        self.metric = metric
        self.kernel_lambda = kernel_lambda
        self.beta = beta
        self.gpr = IncrementalGaussianProcess(kernel_lambda)

    def fit(self, x_queue, y_queue):
        self.gpr.fit(x_queue, y_queue)

    def optimize_acq(self, model_ids, descriptors, timeout):
        start_time = time.time()
        target_graph = None
        father_id = None
        descriptors = deepcopy(descriptors)
        elem_class = Elem
        if self.metric.higher_better():
            elem_class = ReverseElem
        pq = PriorityQueue()
        temp_list = []
        for model_id in model_ids:
            metric_value = self.searcher.get_metric_value_by_id(model_id)
            temp_list.append((metric_value, model_id))
        temp_list = sorted(temp_list)
        for metric_value, model_id in temp_list:
            graph = self.searcher.load_model_by_id(model_id)
            graph.clear_operation_history()
            graph.clear_weights()


@total_ordering
class Elem:
    def __init__(self, metric_value, father_id, graph):
        self.father_id = father_id
        self.graph = graph
        self.metric_value = metric_value

    def __eq__(self, other):
        return self.metric_value == other.metric_value

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class ReverseElem(Elem):
    def __lt__(self, other):
        return self.metric_value > other.metric_value


def contain(descriptors, target_descriptor):
    for descriptor in descriptors:
        if edit_distance(descriptor, target_descriptor, 1) < 1e-5:
            return True
    return False
