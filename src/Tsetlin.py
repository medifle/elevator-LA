from constant import Constant
from core import environment
from random import randint
from typing import List


class Tsetlin:
    def __init__(self):
        self.actions = Constant.ACTIONS.value
        self.experiments = Constant.EXPERIMENTS.value
        self.time_avg_range = Constant.TIME_AVG_RANGE.value
        self.training_times = Constant.TRAINING_TIMES.value
        self.state_depth = Constant.STATE_DEPTH.value
        self.state = randint(1, self.actions * self.state_depth)  # random chosen initial state
        self.targetAccuracy = 0.90  # for speed test

    def got_reward(self, state: int) -> int:
        if state % self.state_depth != 1:
            state -= 1
        return state

    def got_penalty(self, state: int) -> int:
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

    def get_action(self, state: int) -> int:
        if state % self.state_depth == 0:
            return state // self.state_depth
        return state // self.state_depth + 1

    def train(self) -> List[int]:
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
        return [r / (self.experiments * (self.time_avg_range // 100)) for r in result]


if __name__ == '__main__':
    # test
    la = Tsetlin()
    print(la.train())
