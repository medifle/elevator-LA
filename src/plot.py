import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_two_bar(scheme: str, y1: List[float], y2: List[float], label_name: str, n1: float, n2: float,
                 precision: int = 0) -> None:
    """
    # y1 = [75.54, 9.395, 4.424, 3.713, 3.439, 1.389]
    # y2 = [98.033, 0.743, 0.341, 0.276, 0.254, 0.253]
    """
    plt.clf()
    bar_width = .8
    gap = 0.15

    x1 = [1, 4, 7, 10, 13, 16]
    x2 = [x + bar_width + gap for x in x1]

    plt.bar(x1, y1, color="#646FFB", label="{} = {:.{}f}".format(label_name, n1, precision))
    plt.bar(x2, y2, color="#EF553B", label="{} = {:.{}f}".format(label_name, n2, precision))

    plt.xticks([r + 0.5 for r in x1], [1, 2, 3, 4, 5, 6])
    plt.yticks(np.arange(0, max(max(y1), max(y2)) + 10, 10))

    plt.title('{} ensemble average'.format(scheme))
    plt.xlabel('Action')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_bar(scheme: str, y: List[float], label_name: str, label_value: float, precision: int = 4) -> None:
    plt.clf()
    x = [1, 2, 3, 4, 5, 6]
    plt.bar(x, y, color="#646FFB", label="{} = {:.{}f}".format(label_name, label_value, precision))

    plt.xticks(x)
    plt.yticks(np.arange(0, max(y) + 10, 10))

    plt.title('%s ensemble average' % scheme)
    plt.xlabel('Action')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
