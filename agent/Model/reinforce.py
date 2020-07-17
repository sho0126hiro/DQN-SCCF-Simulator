# Standard Import
import os
import sys
from typing import NamedTuple, Tuple, List

# ThirdParty Import 
import numpy as np
import torch.nn as nn

# Internal Import
sys.path.append(os.path.join(os.path.dirname(__file__), '../')) 
from Model.model import AbstractModel
from Model.memory import SimpleMemory
from config import AgentParameter as AP

class Transition(NamedTuple):
    """
    遷移に関する情報を格納する
    """
    state: List[int]
    action: int
    reward: int
    next_state: List[int]

class ModelParameter:
    beta: float = 15.0 # softmax - inverse temperature coefficcient
    eta: float = 0.22  # learning rate
    init_theta: float = 10.0

class REINFORCE(AbstractModel):
    """
    REINFORCE Algorithm Model
    """
    def __init__(self):
        self.memory: SimpleMemory = SimpleMemory[Transition](AP.BATCH_SIZE)
        self.transition: type = Transition
        self.init_tuple = tuple(AP.C for _ in range(AP.T+1)) # (AP.C AP.C , .. AP.C)を生成(len: AP.T)
        self.__theta: np.ndarray = np.full(self.init_tuple, ModelParameter.init_theta)

    def _softmax(self, theta, state: List[int]) -> np.ndarray:
        """
        特定の状態に対するpiを返す
        """
        exp_theta = np.exp(ModelParameter.beta*theta[tuple(state)])
        pi = exp_theta / np.nansum(exp_theta)
        return pi
    
    def _get_pi(self) -> np.ndarray:
        """
        全ての状態でのpiを返す
        """
        # 全状態パタンの生成
        # [0,0,..., 0]　〜
        # [C,C,..., C]
        pattern: List = []
        for i in range(AP.T):
            tmp = []
            for j in range(AP.C**AP.T):
                tmp.append(int(j/AP.C**(AP.T-i-1))%AP.C)
            pattern.append(tmp)
    
        """
        全状態行動パターンの出力
        for i in range(AP.T+1):
            tmp = []
            for j in range(AP.C**(AP.T+1)):
                if(i == AP.T):
                    tmp.append(int(j%AP.C))
                else:
                    tmp.append(int(j/AP.C**(AP.T-i))%AP.C)
            pattern.append(tmp)
        """

        pattern = np.array(pattern).T
        pi = np.empty(self.init_tuple)
        for e in pattern:
            pi[tuple(e)] = self._softmax(self.__theta, e)
        
        return pi

    def get_action(self, state: List[int]) -> int:
        target_prob = self._softmax(self.__theta, state)
        action:int  = np.random.choice([i for i in range(AP.C)], p=target_prob)
        return action

    def update(self):
        memory: List[Transition]= list(self.memory.get_all())

        # 各状態における累計報酬値（ベースラインで使用する）
        cumlative_rewards: np.ndarray = np.zeros(tuple(AP.C for _ in range(AP.T)))
        # 累計報酬を加算　-> baselineに使用
        for t in memory:
            cumlative_rewards[tuple(t.state)] += t.reward
        # 更新式のSigmaの中の累計
        sum_: np.ndarray = np.zeros(self.init_tuple)
        
        pi = self._get_pi()
        
        for t in memory: # type: Transitiond
            sa = t.state + [t.action]
            tmp: float = 1 - pi[tuple(sa)]
            baseline: float = cumlative_rewards[tuple(t.state)] / AP.BATCH_SIZE
            tmp2: float = t.reward - baseline
            sum_[tuple(sa)] += tmp * tmp2

        delta_j = sum_ / AP.BATCH_SIZE
        self.__theta += ModelParameter.eta * delta_j
    
    def add_memory(self, t: Transition):
        return self.memory.add(t)
    
    def build_transition(self, state: List[int], action: int, reward: int, next_state: List[int]) -> Transition:
        return Transition(state, action, reward, next_state)
    
    def get_modelname(self) -> str:
        return self.__class__.__name__
