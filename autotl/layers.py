from tensorlayer.layers import *


def is_layer(layer, layer_type):
    if layer_type == 'Input':
        return isinstance(layer, StubInput)
    if layer_type == 'Conv':
        return isinstance(layer, StubConv)
    if layer_type == 'Dense':
        return isinstance(layer, (StubDense,))
    if layer_type == 'BatchNormalization':
        return isinstance(layer, (StubBatchNormalization,))
    if layer_type == 'Concatenate':
        return isinstance(layer, (StubConcatenate,))
    if layer_type == 'Add':
        return isinstance(layer, (StubAdd,))
    if layer_type == 'Pooling':
        return isinstance(layer, StubPooling)
    if layer_type == 'Dropout':
        return isinstance(layer, (StubDropout,))
    if layer_type == 'Softmax':
        return isinstance(layer, (StubSoftmax,))
    if layer_type == 'ReLU':
        return isinstance(layer, (StubReLU,))
    if layer_type == 'Flatten':
        return isinstance(layer, (StubFlatten,))
    if layer_type == 'GlobalAveragePooling':
        return isinstance(layer, StubGlobalPooling)


class StubLayer(object):
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


class StubWeightBiasLayer(StubLayer):
    def import_weights(self, tl_layer):
        self

class StubBatchNormalization(StubWeightBiasLayer):
    pass

class StubDense(StubWeightBiasLayer):
    pass

class StubConv(StubWeightBiasLayer):
    pass

class StubAggregateLayer(StubLayer):
    pass

class StubConcatenate(StubAggregateLayer):
    pass

class StubAdd(StubAggregateLayer):
    @property
    def output_shape(self):
        return self.input[0].shape


class StubFlatten(StubLayer):
    @property
    def output_shape(self):
        ret = 1
        for dim in self.input.shape:
            ret *= dim
        return ret,

class StubReLU(StubLayer):
    pass


class StubSoftmax(StubLayer):
    pass


class StubPooling(StubLayer):
    def __init__(self, kernel_size=2, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.kernel_size = kernel_size

    @property
    def output_shape(self):
        ret = tuple()
        for dim in self.input.shape[:-1]:
            ret = ret + (max(int(dim / self.kernel_size), 1),)
        ret = ret + (self.input.shape[-1],)
        return ret


class StubGlobalPooling(StubLayer):
    def __init__(self, func, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.func = func


class StubDropout(StubLayer):
    def __init__(self, rate, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.rate = rate


class StubInput(StubLayer):
    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
