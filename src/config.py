from enum import Enum


class Constant(Enum):
    # Q = [1, 2, 3, 4, 5, 6]
    Q = [6, 5, 4, 3, 2, 1]
    ACTIONS = 6
    EXPERIMENTS = 100  # 100
    TRAINING_TIMES = 11000
    TIME_AVG_RANGE = 1000

    # FSSA
    GAUSS_SIGMA = 1
    STATE_DEPTH = 6

    # VSSA: Lri
    LEARNING_PARAM = 0.02  # 0 < λ < 1

    # speed test
    TERMINAL_ACCURACY = 0.9  # termination threshold
    ACCURACY_RECORD_RANGE = 50
