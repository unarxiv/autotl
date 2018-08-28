import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def layer_distance(a, b):
    return abs(a - b) * 1.0 / max(a, b)


def layers_distance(list_a, list_b):
    len_a = len(list_a)
    len_b = len(list_b)
    f = np.zeros((len_a + 1, len_b + 1))
    f[-1][-1] = 0
    for i in range(-1, len_a):
        f[i][-1] = i + 1
    for j in range(-1, len_b):
        f[-1][j] = j + 1
    for i in range(len_a):
        for j in range(len_b):
            f[i][j] = min(
                f[i][j - 1] + 1, f[i - 1][j] + 1,
                f[i - 1][j - 1] + layer_distance(list_a[i], list_b[j]))
    return f[len_a - 1][len_b - 1]


def vector_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def skip_connection_distance(a, b):
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / (
        max(a[0], b[0]) + max(len_a, len_b))


def skip_connections_distance(list_a, list_b):
    distance_matrix = np.zeros((len(list_a), len(list_b)))
    for i, a in enumerate(list_a):
        for j, b in enumerate(list_b):
            distance_matrix[i][j] = skip_connection_distance(a, b)
    return distance_matrix[linear_sum_assignment(distance_matrix)].sum() + abs(
        len(list_a) - len(list_b))


def edit_distance(x, y, kernel_lambda):
    ret = 0
    ret += layers_distance(x.conv_widths, y.conv_widths)
    ret += layers_distance(x.dense_widths, y.dense_widths)
    ret += kernel_lambda * skip_connections_distance(x.skip_connections,
                                                     y.skip_connections)
    return ret


def edit_distance_matrix(kernel_lambda, train_x, train_y=None):
    if train_y is None:
        ret = np.zeros((train_x.shape[0], train_x.shape[0]))
        for x_index, x in enumerate(train_x):
            for y_index, y in enumerate(train_x):
                if x_index == y_index:
                    ret[x_index][y_index] = 0
                elif x_index < y_index:
                    ret[x_index][y_index] = edit_distance(x, y, kernel_lambda)
                else:
                    ret[x_index][y_index] = ret[y_index][x_index]
        return ret
    ret = np.zeros((train_x.shape[0], train_y.shape[0]))
    for x_index, x in enumerate(train_x):
        for y_index, y in enumerate(train_y):
            ret[x_index][y_index] = edit_distance(x, y, kernel_lambda)
    return ret


def bourgain_embedding_matrix(distance_matrix):
    distance_matrix = np.array(distance_matrix)
    n = len(distance_matrix)
    if n == 1:
        return distance_matrix
    np.random.seed(123)
    distort_elements = []
    r = range(n)
    k = int(math.ceil(math.log(n) / math.log(2) - 1))
    t = int(math.ceil(math.log(n)))
    counter = 0
    for i in range(0, k + 1):
        for t in range(t):
            s = np.random.choice(r, 2**i)
            for j in r:
                d = min([distance_matrix[j][s] for s in s])
                counter += len(s)
                if i == 0 and t == 0:
                    distort_elements.append([d])
                else:
                    distort_elements[j].append(d)
    distort_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distort_matrix[i][j] = distort_matrix[j][i] = vector_distance(
                distort_elements[i], distort_elements[j])
    return np.array(distort_matrix)
