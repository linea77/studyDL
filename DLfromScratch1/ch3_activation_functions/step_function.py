import numpy as np
import matplotlib.pylab as plt


def step_function_float(self, x: float):
    if x > 0:
        return 1
    else:
        return 0


def step_function_np_array(self, x: np.array):
    y = x > 0  # ex. y = array([True, False, False], dtype=bool)
    return y.astype(np.int32)  # transform bool array -> int array


def draw_step_function_graph(self):
    x = np.arange(-0.5, 0.5, 0.1)
    y = self.input_np_array(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__ == "__main__":
    x = np.array([-1.0, 1.0, 2.0])
    print(step_function_np_array(x))
    draw_step_function_graph()
