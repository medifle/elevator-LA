from math import ceil
from random import gauss
import sys


def environment(action: int) -> int:
    if action > 6 or action < 1:
        print("list index out of range error. i =", action)
        sys.exit()

    Q = [1, 2, 3, 4, 5, 6]

    def f(i: int) -> int:
        h = gauss(0, 1)  # 1 is sigma
        fi = 0.8 * Q[i - 1] + 0.4 * ceil(Q[i - 1] / 2) + h
        if fi < 0:
            fi = 0
        return fi

    fi_list = [f(x) for x in range(1, 7)]
    min_fi = min(fi_list)
    # print(fi_list, fi_list.index(min_fi))  # test
    if fi_list[action - 1] == min_fi:
        return 0
    else:
        return 1


if __name__ == '__main__':
    # test
    for e in range(50):
        arr = []

        arr.append(environment(1))
        # for i in range(1, 7):
        #     arr.append(environment(i))

        print(arr)
