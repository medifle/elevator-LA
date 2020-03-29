from config import Constant
from core import got_penalty_tsetlin, got_reward_tsetlin, train_fssa, get_action_fssa, speed_test_fssa, \
    plot_group_bar_fssa
from random import randint
from typing import List


class Tsetlin:
    def __init__(self):
        self.actions = Constant.ACTIONS.value
        self.experiments = Constant.EXPERIMENTS.value
        self.time_avg_range = Constant.TIME_AVG_RANGE.value
        self.training_times = Constant.TRAINING_TIMES.value
        self.state_depth = Constant.TRAINING_TIMES.value
        self.state = randint(1, self.actions * self.state_depth)  # random chosen initial state

    def reset_state(self):
        self.state = randint(1, self.actions * self.state_depth)

    def got_reward(self, state: int) -> int:
        return got_reward_tsetlin(self, state)

    def got_penalty(self, state: int) -> int:
        return got_penalty_tsetlin(self, state)

    def get_action(self, state: int) -> int:
        return get_action_fssa(self, state)

    def train(self, **kwargs: int) -> List[int]:
        n = kwargs.get('n', None)
        if n:
            self.state_depth = n
        return train_fssa(self)

    def speed_test(self) -> int:
        return speed_test_fssa(self)

    def plot_group_bar(self, state_depth: int) -> None:
        plot_group_bar_fssa(self, state_depth)


if __name__ == '__main__':
    # test
    la = Tsetlin()
    # print(la.train())
    # la.speed_test()
    la.plot_group_bar(50)
