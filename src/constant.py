from enum import Enum


class Constant(Enum):
    Q = [1, 2, 3, 4, 5, 6]
    ACTIONS = 6
    EXPERIMENTS = 100  # 100
    TRAINING_TIMES = 10000
    TIME_AVG_RANGE = 1000
    GAUSS_SIGMA = 1.7  # 1.8
    STATE_DEPTH = 6  # FSSA
    LEARNING_PARAM = 0.0018  # Lri
    TARGET_ACCURACY = 0.9  # speed test termination threshold
