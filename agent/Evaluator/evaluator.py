# Standard Import 
from typing import List
from abc import ABCMeta, abstractmethod
# ThirdParty Import
import numpy as np

class AbstractEvaluator(metaclass=ABCMeta):
    """
    評価器はこれを継承し、abstractmethodを必ず実装しなければならない
    """

    @abstractmethod
    def evaluate(self, state, action):
        pass
