import numpy as np
from autotl.tasks.cv.constant import Constant


class OneHotEncoder(object):
    def __init__(self):
        self.data = None
        self.n_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        data = np.array(data).flatten()
        self.labels = set(data)
        self.n_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.n_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        data = np.array(data)
        if len(data.shape) < 1:
            data = data.flatten()
        return np.array(list(map(lambda x:self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        return np.array(list(map(lambda x:self.int_to_label[x], np.argmax(np.array(data), axis=1))))

class DataTransformer:
    def __init__(self, data, augment=Constant.DATA_AUGMENTATION):
        self.max_val = data.max()
        data = data / self.max_val
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.augment = augment

    def transform_train(self, data, targets=None, batch_size=None):
        if not self.augment:
            augment_list = []
        else:
            augment_list = []


def validate(x_train, y_train):
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) < 2:
        raise ValueError('x_train should has at least 2 dims.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            'x_train and y_train should have the same number of instances.')
