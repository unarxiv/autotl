import os
import numpy as np
from abc import abstractmethod
from autotl.fileutils import temp_folder_generator
from autotl.fileutils import has_file
from autotl.fileutils import pickle_from_file
from autotl.fileutils import ensure_dir
from autotl.tasks.supervised import Supervised
from autotl.tasks.cv.preprocessor import validate
from autotl.loss import classification_loss
from autotl.metric import Accuracy
from autotl.tasks.cv.preprocessor import OneHotEncoder
from constant import Constant


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