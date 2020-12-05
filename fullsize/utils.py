import numpy as np


def rescale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    x = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    x = x * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return x