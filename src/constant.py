from enum import Enum


class Constant(Enum):
    ACTIONS = 6
    EXPERIMENTS = 100  # 100
    TRAINING_TIMES = 10000
    TIME_AVG_RANGE = 1000
    GAUSS_SIGMA = 10  # 1.5
    STATE_DEPTH = 6  # FSSA
    LEARNING_PARAM = 0.02  # Lri
    TARGET_ACCURACY = 0.9  # speed test termination threshold
