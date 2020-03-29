from math import ceil
from random import gauss, random
from typing import List
from config import Constant
from collections import deque
import sys


def answer_with_probability(p: float) -> bool:
    r = random()
    if r < p:
        return True
    return False


def environment(action: int) -> int:
    if action > 6 or action < 1:
        print("list index out of range error. i =", action)
        sys.exit()

    Q = Constant.Q.value

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
    print("%s: start training..." % self.__class__.__name__)
    accuracy_list = [0 for i in range(self.actions)]
    for e in range(self.experiments):
        self.reset_state()
        for t in range(self.training_times):
            action = self.get_action(self.state)
            beta = environment(action)

            # print('{:>2}'.format(self.state), action, beta)  # test
            if beta == 1:  # penalty
                self.state = self.got_penalty(self.state)
            else:  # reward
                self.state = self.got_reward(self.state)
            if t > (self.training_times - self.time_avg_range):
                accuracy_list[action - 1] += 1
    print("%s: finished" % self.__class__.__name__)
    return [r / (self.experiments * (self.time_avg_range // 100)) for r in accuracy_list]


def train_vssa(self) -> List[int]:
    print("%s: start training..." % self.__class__.__name__)
    accuracy_list = [0 for i in range(self.actions)]
    for e in range(self.experiments):
        self.reset_state()
        for t in range(self.training_times):
            action = self.get_action()
            beta = environment(action)

            # print(self.state, action, beta)  # test
            if beta == 0:
                self.got_reward(action)

            if t > (self.training_times - self.time_avg_range):
                accuracy_list[action - 1] += 1
    print("%s: finished" % self.__class__.__name__)
    return [r / (self.experiments * (self.time_avg_range // 100)) for r in accuracy_list]


def not_terminated(action_stat: List[int]) -> bool:
    accuracy_calc_range = Constant.ACCURACY_RECORD_RANGE.value
    terminal_accuracy = Constant.TERMINAL_ACCURACY.value
    for e in action_stat:
        if e / accuracy_calc_range > terminal_accuracy:
            return False
    return True


def speed_test_fssa(self) -> int:
    print("%s: speed test starts..." % self.__class__.__name__)
    time_to_converge_sum = 0
    accuracy_record_range = Constant.ACCURACY_RECORD_RANGE.value
    for e in range(self.experiments):
        self.reset_state()
        queue = deque([])
        action_stat = [0 for i in range(self.actions)]

        time_to_converge = 0
        while not_terminated(action_stat):
            time_to_converge += 1

            action = self.get_action(self.state)
            beta = environment(action)

            # print('{:>2}'.format(self.state), action, beta, queue)  # test
            if beta == 1:  # penalty
                self.state = self.got_penalty(self.state)
            else:  # reward
                self.state = self.got_reward(self.state)

            # maintain queue and action_stat
            if len(queue) > accuracy_record_range:
                poped_action = queue.popleft()
                action_stat[poped_action - 1] -= 1
            queue.append(action)
            action_stat[action - 1] += 1
        time_to_converge_sum += time_to_converge
    print("%s: speed test finished" % self.__class__.__name__)
    result = time_to_converge_sum // self.experiments
    print("%s: time to converge" % self.__class__.__name__, result)
    return result


def speed_test_vssa(self) -> int:
    print("%s: speed test starts..." % self.__class__.__name__)
    time_to_converge_sum = 0
    accuracy_record_range = Constant.ACCURACY_RECORD_RANGE.value
    for e in range(self.experiments):
        self.reset_state()
        queue = deque([])
        action_stat = [0 for i in range(self.actions)]

        time_to_converge = 0
        while not_terminated(action_stat):
            time_to_converge += 1

            action = self.get_action()
            beta = environment(action)

            # print(self.state, action, beta, queue)  # test
            if beta == 0:
                self.got_reward(action)

            # maintain queue and action_stat
            if len(queue) > accuracy_record_range:
                poped_action = queue.popleft()
                action_stat[poped_action - 1] -= 1
            queue.append(action)
            action_stat[action - 1] += 1
        time_to_converge_sum += time_to_converge
    print("%s: speed test finished" % self.__class__.__name__)
    result = time_to_converge_sum // self.experiments
    print("%s: time to converge" % self.__class__.__name__, result)
    return result


if __name__ == '__main__':
    # test: environment C_i distribution
    countArr = [0, 0, 0, 0, 0, 0]
    iterations = 1000
    for e in range(iterations):
        arr = []
        for i in range(1, 7):
            arr.append(environment(i))
        print(arr)
        for j in range(6):
            if arr[j] == 0:
                countArr[j] += 1
    print([countSum / iterations for countSum in countArr])
