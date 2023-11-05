import numpy as np
from typing import Union

def relu(x):
    # if x < 0:
    #     return 0
    # else :
    #     return x
    return np.maximum(0, x)


def sigmoid(x: Union[float, np.array]):
    return 1 / (1 + np.exp(-x))



def softmax(a:np.array):
    c = np.max(a)
    exp_a = np.exp(a - c)           # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def step_function_np_array(self, x: np.array):
    y = x > 0  # ex. y = array([True, False, False], dtype=bool)
    return y.astype(np.int32)  # transform bool array -> int array
