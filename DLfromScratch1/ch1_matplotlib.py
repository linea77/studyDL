import numpy as np
import matplotlib.pyplot as plt


def sin_plot(x):
    y = np.sin(x)
    plt.plot(x, y, label="sin")


def cos_plot(x):
    y = np.cos(x)
    plt.plot(x, y, linestyle="--", label="cos")


def draw_plot():
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def main():
    x = np.arange(0, 6, 0.1)

    sin_plot(x)
    cos_plot(x)
    draw_plot()


if __name__ == "__main__":
    main()