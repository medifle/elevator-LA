from math import ceil
from random import gauss
from typing import List
from constant import Constant
import sys


def environment(action: int) -> int:
    if action > 6 or action < 1:
        print("list index out of range error. i =", action)
        sys.exit()

    Q = [1, 2, 3, 4, 5, 6]

    def f(i: int) -> int:
        h = gauss(0, Constant.GAUSS_SIGMA.value)
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


def got_reward_tsetlin(self, state: int) -> int:
    """return new current state"""
    if state % self.state_depth != 1:
        state -= 1
    return state


def got_penalty_tsetlin(self, state: int) -> int:
    """return new current state"""
    # if state is max, go back to N (the head of the first line)
    if state == self.state_depth * self.actions:
        state = self.state_depth
    # else if state if the head of a line, jump onto the head of the next line
    elif state % self.state_depth == 0:
        state += self.state_depth
    else:
        state += 1
    return state


def get_action_fssa(self, state: int) -> int:
    if state % self.state_depth == 0:
        return state // self.state_depth
    return state // self.state_depth + 1


def train_fssa(self) -> List[int]:
    print("%s : start training..." % self.__class__.__name__)
    result = [0 for i in range(self.actions)]
    for e in range(self.experiments):
        for t in range(self.training_times):
            action = self.get_action(self.state)
            beta = environment(action)

            # print('{:>2}'.format(self.state), action, beta)  # test

            if beta == 1:  # penalty
                self.state = self.got_penalty(self.state)
            else:  # reward
                self.state = self.got_reward(self.state)
            if t > (self.training_times - self.time_avg_range):
                result[action - 1] += 1
    print("%s : finished" % self.__class__.__name__)
    return [r / (self.experiments * (self.time_avg_range // 100)) for r in result]


if __name__ == '__main__':
    # test
    for e in range(50):
        arr = []

        # arr.append(environment(1))
        # for i in range(1, 7):
        #     arr.append(environment(i))

        # print(arr)
