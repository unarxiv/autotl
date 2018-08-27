from autotl.constant import Constant
class Search(object):
    def __init__(self, n_output_node, input_shape, path, metric, loss, verbose,
                 trainer_args = None,
                 default_model_len = Constant.MODEL_LEN,
                 default_model_width = Constant.MODEL_WIDTH,
                 beta = Constant.BETA,
                 kernel_lambda = Constant.KERNEL_LAMBDA,
                 t_min = Constant.T_MIN):
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
        ret = {
            'name': u,
            'children': children
        }
        return ret
        