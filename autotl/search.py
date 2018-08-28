import logging
import os

from autotl.constant import Constant
from autotl.fileutils import pickle_from_file, pickle_to_file
from autotl.optimizer import BayesianOptimizer


class Searcher(object):
    def __init__(self,
                 n_output_node,
                 input_shape,
                 path,
                 metric,
                 loss,
                 verbose,
                 trainer_args=None,
                 default_model_len=Constant.MODEL_LEN,
                 default_model_width=Constant.MODEL_WIDTH,
                 beta=Constant.BETA,
                 kernel_lambda=Constant.KERNEL_LAMBDA,
                 t_min=Constant.T_MIN):
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.metric = metric
        self.loss = loss
        self.path = path
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER
        self.search_tree = SearchTree()
        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        self.bo = BayesianOptimizer(self, t_min, metric, kernel_lambda, beta)

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id + '.h5')))

    def get_best_model_id(self):
        if self.metric.higher_better():
            return max(
                self.history, key=lambda x: x['metric_value'])['model_id']
        else:
            return min(
                self.history, key=lambda x: x['metric_value'])['model_id']

    def load_best_model(self):
        return self.load_model_by_id(self.get_best_model_id())

    def replace_model(self, graph, model_id):
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.h5'))

    def add_model(self, metric_value, loss, graph, model_id):
        if self.verbose:
            logging.info('\n [AutoTL]: Saving Model...')
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.h5'))
        ret = {
            'model_id': model_id,
            'loss': loss,
            'metric_value': metric_value
        }
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            with open(os.path.join(self.path, 'best_model.txt', 'w')) as f:
                f.write('best model: ' + str(model_id))
                f.close()

        if self.verbose:
            idx = ['model_id', 'loss', 'metric_value']
            header = ['Model ID', 'Loss', 'Metric Value']
            line = '|'.join(x.center(24) for x in header)
            print('+' + '-' * len(line) + '+')
            print('|' + line + '|')
            for i, r in enumerate(self.history):
                print('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                print('|' + line + '|')
            print('+' + '-' * len(line) + '+')

        descriptor = graph.extract_descriptor()
        self.x_queue.append(descriptor)
        self.y_queue.append(metric_value)
        return ret


class SearchTree(object):
    def __init__(self):
        self.root = None
        self.adj_list = {}

    def add_child(self, u, v):
        if u == -1:
            self.root = v
            self.adj_list[v] = []
            return
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if v not in self.adj_list:
            self.adj_list[v] = []

    def get_dict(self, u=None):
        if u is None:
            return self.get_dict(self.root)
        children = []
        for v in self.adj_list[u]:
            children.append(self.get_dict(v))
        ret = {'name': u, 'children': children}
        return ret
