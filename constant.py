class DeepQNetwork:
    LEARNING_RATE: float = 0.01
    GAMMA: float = 0.80

class ReplayMemory:
    CAPACITY: int = 30

class Agent:
    TRY = 3000
    EPISODE = 10
    EPSILON = 0.2
    EPSILON_DECAY = 0.97
    EPSILON_MIN = 0.1
    C = 10 # num of category
    ACTIONS = [i for i in range(C)]
    CHECK = 100

class Eavluator:
    EPSILON = 0.3

class Reward_A:
    GOOD = 1
    BAD = 1
    IGNORE = 0

class Reward_B:
    GOOD = 1
    BAD = -1
    IGNORE = 0
