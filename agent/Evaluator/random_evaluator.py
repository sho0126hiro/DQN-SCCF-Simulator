# Standard Import
import random
from typing import List

# ThirdPary Import
import numpy as np

# Internal Import
from Evaluator.evaluator import AbstractEvaluator
from config import AgentParameter as AP

class RandomEvaluator(AbstractEvaluator):
    
    def evaluate(self, state: List[int], action: int):
        """
        ランダム評価を行う
        """
        return random.sample(AP.REWARD, 1)[0]
    