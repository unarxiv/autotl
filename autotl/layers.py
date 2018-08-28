from tensorlayer.layers import *


class StubLayer:
    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node
        self.weights = None

    def build(self, shape):
        pass

    def set_weights(self, weights):
        self.weights = weights

    def import_weights(self, tl_layer):
        pass

    def export_weights(self, tl_layer):
        pass

    def get_weights(self):
        return self.weights

    @property
    def output_shape(self):
        return self.input.shape
