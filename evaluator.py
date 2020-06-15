import numpy as np
import constant

class ProbEvaluator:
    """
    特定の確率で評価する
    e-greedyのような感じで、eの確率で、完全ランダム選択を行う
    """
    def __init__(self):
        self.preference = [0.4, 0.3, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0]
        self.rewards = [1,0]

    def evaluate(self, state: int, action: int) -> float:
        if np.random.rand() < constant.Eavluator.EPSILON:
            return np.random.choice(self.rewards)
        else:
            prob = [self.preference[action]*2, 1-self.preference[action]*2]
            return np.random.choice(a = self.rewards, p = prob)
        