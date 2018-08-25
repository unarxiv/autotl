import random
import numpy as np


def layer_distance(a, b):
    return abs(a-b) * 1.0 / max(a,b)

def layers_distance(a, b):
    len_a = len(a)
    len_b = len(b)
    f = np.zeros((len_a+1, len_b+1))
    