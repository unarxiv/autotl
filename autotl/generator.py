from abc import ABC, abstractmethod

from autotl.constant import Constant
from autotl.graph import Graph


class Generator(object):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, model_len, model_width):
        pass

class CNNGenerator(Generator):
    def __init__(self, n_output_node, input_shape):
        self.n_output_node = n_output_node
        self.input_shape = input_shape
        if len(self.input_shape) > 4:
            raise ValueError("The Input Dimension is too high")
        if len(self.input_shape) < 2:
            raise ValueError("The Input Dimension is too low")
            
    def generate(self, model_len=Constant.MODEL_LEN, model_width=Constant.MODEL_WIDTH):
        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        for i in range(model_len):
