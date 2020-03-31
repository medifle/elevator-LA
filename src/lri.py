from config import Constant
from core import train_vssa, speed_test_vssa
from random import uniform
from plot import plot_bar, plot_two_bar
from typing import List


class Lri:
    def __init__(self):
        self.actions = Constant.ACTIONS.value
        self.experiments = Constant.EXPERIMENTS.value
        self.time_avg_range = Constant.TIME_AVG_RANGE.value
        self.training_times = Constant.TRAINING_TIMES.value
        self.state = [1 / self.actions for i in range(self.actions)]  # action probability vector
        self.a = Constant.LEARNING_PARAM.value

    def reset_state(self):
        self.state = [1 / self.actions for i in range(self.actions)]

    def got_reward(self, action: int) -> None:
        state_length = len(self.state)
        for i in range(1, state_length + 1):
            if i == action:
                self.state[i - 1] = self.state[i - 1] + self.a * (1 - self.state[i - 1])
            else:
                self.state[i - 1] = (1 - self.a) * self.state[i - 1]

    def get_action(self) -> int:
        """choose action by action probability distribution"""
        r = uniform(0, sum(self.state))
        prob_accumulation = 0
        state_length = len(self.state)
        for i in range(state_length):
            prob_accumulation += self.state[i]
            if prob_accumulation > r:
                return i + 1

    def train(self, **kwargs: float) -> List[int]:
        lp = kwargs.get('lp', None)
        if lp:
            self.a = lp
        return train_vssa(self)

    def speed_test(self, best_action: int) -> int:
        return speed_test_vssa(self, best_action)

    def plot_bar(self) -> None:
        y = self.train()
        print(y)
        plot_bar(self.__class__.__name__, y, "λ", self.a)

    def plot_two_bar(self, lp: float) -> None:
        y1 = self.train()
        print(y1)
        y2 = self.train(lp=lp)
        print(y2)
        plot_two_bar(self.__class__.__name__, y1, y2, "λ", Constant.LEARNING_PARAM.value, lp, 4)


if __name__ == '__main__':
    # test
    la = Lri()
    print(la.train())
    # la.speed_test(1)
    # la.plot_two_bar(0.5000)
