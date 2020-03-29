from config import Constant
from core import got_penalty_tsetlin, train_fssa, get_action_fssa, answer_with_probability, got_reward_tsetlin, \
    speed_test_fssa
from random import randint
from typing import List


class Krylov:
    def __init__(self):
        self.actions = Constant.ACTIONS.value
        self.experiments = Constant.EXPERIMENTS.value
        self.time_avg_range = Constant.TIME_AVG_RANGE.value
        self.training_times = Constant.TRAINING_TIMES.value
        self.state_depth = Constant.STATE_DEPTH.value
        self.state = randint(1, self.actions * self.state_depth)  # random chosen initial state

    def reset_state(self):
        self.state = randint(1, self.actions * self.state_depth)

    def got_reward(self, state: int) -> int:
        return got_reward_tsetlin(self, state)

    def got_penalty(self, state: int) -> int:
        if answer_with_probability(0.5):
            return got_penalty_tsetlin(self, state)
        else:
            return got_reward_tsetlin(self, state)

    def get_action(self, state: int) -> int:
        return get_action_fssa(self, state)

    def train(self) -> List[int]:
        return train_fssa(self)

    def speed_test(self) -> int:
        return speed_test_fssa(self)


if __name__ == '__main__':
    # test
    la = Krylov()
    print(la.train())
    # la.speed_test()
