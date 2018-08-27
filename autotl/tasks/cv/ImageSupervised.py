import os
import numpy as np
from functools import reduce
from abc import abstractmethod

from sklearn.metrics import accuracy_score

from autotl.fileutils import has_file
from autotl.fileutils import ensure_dir
from autotl.fileutils import pickle_to_file
from autotl.fileutils import pickle_from_file
from autotl.fileutils import temp_folder_generator

from autotl.tasks.supervised import Supervised
from autotl.tasks.cv.constant import Constant
from autotl.tasks.cv.preprocessor import validate
from autotl.tasks.cv.preprocessor import OneHotEncoder

from autotl.loss import classification_loss
from autotl.metric import Accuracy


class ImageSupervised(Supervised):
    def __init__(self,
                 verbose=False,
                 path=None,
                 resume=False,
                 searcher_args=None,
                 augment=None):
        super().__init__(verbose)
        if searcher_args is None:
            searcher_args = {}
        if path is None:
            path = temp_folder_generator()
        if augment is None:
            augment = Constant.DATA_AUGMENTATION
        if has_file(os.path.join(path, 'classifier')) and resume:
            classifier = pickle_from_file(os.path.join(path, 'classifier'))
            self.__dict__ = classifier.__dict__
            self.path = path
        else:
            self.y_encoder = None
            self.data_transformer = None
            self.verbose = verbose
            self.searcher = False
            self.path = path
            self.searcher_args = searcher_args
            self.augment = augment
            ensure_dir(path)

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    def fit(self, x_train=None, y_train=None, time_limit=None):
        if y_train is None:
            y_train = []
        if x_train is None:
            x_train = []
        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()
        validate(x_train, y_train)
        y_train = self.transform_y(y_train)

    def transform_y(self, y_train):
        return y_train
    
    def predict(self, x_test):
        if Constant.LIMIT_MEMORY:
            pass
        test_loader = self.data_transformer.transform_test(x_test)
        model = self.load_searcher().load_best_model().produce_model()
        model.eval()
        outputs = []
        for index, inputs in enumerate(test_loader):
            outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x,y)), outputs)
        return self.inverse_transform_y(output)

    def inverse_transform_y(self, output):
        return output

    def load_searcher(self):
        return pickle_from_file(os.path.join(self.path, 'searcher'))
    
    def save_searcher(self, searcher):
        pickle_to_file(searcher, os.path.join(self.path, 'searcher'))
    
    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)


class ImageClassifier(ImageSupervised):

    @property
    def loss(self):
        return classification_loss

    def transform_y(self, y_train):
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)
        return y_train

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def get_n_output_node(self):
        return self.y_encoder.n_classes

    @property
    def metric(self):
        return Accuracy
