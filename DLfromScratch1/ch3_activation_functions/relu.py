import numpy as np


def relu(x):
    # if x < 0:
    #     return 0
    # else :
    #     return x
    return np.maximum(0, x)
