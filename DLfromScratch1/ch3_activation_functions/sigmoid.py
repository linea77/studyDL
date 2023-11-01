import numpy as np
import matplotlib.pylab as plt
from typing import Union


def sigmoid(x: Union[float, np.array]):
    return 1 / (1 + np.exp(-x))


def draw_sigmoid_graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__ == "__main__":
    x = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(x))
    draw_sigmoid_graph()
