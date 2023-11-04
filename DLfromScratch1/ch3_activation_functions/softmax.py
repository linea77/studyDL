import numpy as np


def softmax(a:np.array):
    c = np.max(a)
    exp_a = np.exp(a - c)           # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y