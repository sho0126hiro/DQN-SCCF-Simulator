# Standard Import 
import sys
from typing import List, NamedTuple
import random
import copy

# ThirdParty Import
import numpy as np

#internal Imports
from Model.model import AbstractModel
from Model.memory import ReplayMemory
from Model.dqn import DQN, Transition
from Model.dqn import ModelParameter as DQNMP
from config import AgentParameter as AP

class ModelParameter:
    dqn_update_count: int = 3
    epoch: int = 3

class DDQN(AbstractModel):
    """
    Double Deep-Q-Network
    """
    def __init__(self):
        self.actDQN = DQN() # 行動選択用ネットワーク こちらを学習対象 mainQN
        self.evalDQN = DQN() # 価値計算用ネットワーク 定期的にself.actDQNからコピーされる
        self.update_count = 0
    
    def get_action(self, state: List[int])-> int:
        return self.actDQN.get_action(state)
    
    def update(self):
        if self.update_count % ModelParameter.dqn_update_count == 0:
            # 定期的にevalDQNに行動選択用ネットワークをコピーする
            self.evalDQN = copy.deepcopy(self.actDQN)
        
        for i in range(ModelParameter.epoch):
            batch_size: float = min(DQNMP.mini_batch_size, len(self.actDQN.replay_memory))
            minibatch : List[Transition] = self.actDQN.replay_memory.sample(batch_size)
            target_data: List[float] = [] # 教師データ
            # build target data
            for element in minibatch:
                next_q_max: float = -sys.float_info.max
                next_action: int = 0
                # 行動選択・Q値の評価(DDQNではここが分離される)
                for next_move in range(AP.C):
                    next_q: float = self.actDQN._predict(element.next_state, next_move)
                    if next_q_max < next_q:
                        # next_q_max = next_q
                        next_action = next_move
                # 評価用Networkを使用して次の行動を評価
                next_q = self.evalDQN._predict(element.next_state, next_action)
                target: float = element.reward + DQNMP.gamma * next_q
                target_data.append(target)
        self.update_count += 1

    def add_memory(self, t: Transition):
        self.actDQN.add_memory(t)
    
    def build_transition(self,state,action,reward,next_state):
        return self.actDQN.build_transition(state, action, reward, next_state)

    def get_modelname(self) -> str:
        return self.__class__.__name__
    
    def eval(self):
        self.actDQN.model.eval()
