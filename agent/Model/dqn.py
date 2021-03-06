# Standard Import 
import sys
from typing import List, NamedTuple
import random

# ThirdParty Import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#internal Imports
from Model.model import AbstractModel
from Model.memory import ReplayMemory
from config import AgentParameter as AP

class Transition(NamedTuple):
    state: List[int] # 2回前までにカテゴりの配列 [1step before, 2step before]
    action: int # 次に発話する発話カテゴリ番号
    reward: int # 0 or 1 or -1
    next_state: List[int] # 次の状態 [action, 1step before]

class ModelParameter:
    alpha: float = 0.01 # learning Rate
    memory_capacity: int = 100
    gamma: float = 0.1 # discount rate
    mini_batch_size: int = 50 # experience replay minibatch
    input_size: int = AP.T + 1
    output_size: int = 1
    # epsilon-greadyでmaxを取得する
    # epsilonは指数関数的に減少させる
    epsilon_start = 0.2
    epsilon_decay = 1000
    epsilon_end = 0.2
    epoch: int = 5

class DQN(AbstractModel):
    """
    Deep-Q-Networkのモデル
    """
    def __init__(self):
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(ModelParameter.input_size,50))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(50,50))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(50, ModelParameter.output_size))
        self.optimiser = optim.Adam(self.model.parameters(), 
                                    lr = ModelParameter.alpha)
        # self.criterion = nn.MSELoss()
        self.criterion = F.smooth_l1_loss
        self.replay_memory = ReplayMemory[Transition](ModelParameter.memory_capacity)
        self.select_action_count = 0

    def _build_input(self, state: List[int], action: int) -> torch.Tensor:
        """
        NNの入力を生成する
        """
        i: List[int] = []
        i.extend(state)
        i.append(action)
        return torch.Tensor(i)

    def _predict(self, state: List[int], action: int) -> float:
        """
        * self.model.eval()されている状態で入ること
        """
        self.optimiser.zero_grad()
        in_: torch.Tensor = self._build_input(state, action)
        q_value: float = float(self.model(in_))
        return q_value

    def get_action(self, state:List[int]) -> int:
        """
        行動の出力
        epsilon-greedy, epsilonを指数関数的に減少させる
        """
        epsilon: float = random.random()
        epsilon_threshold = ModelParameter.epsilon_end + (ModelParameter.epsilon_start - ModelParameter.epsilon_end) * np.exp(-1. * self.select_action_count / ModelParameter.epsilon_decay)
        self.select_action_count += 1
        if epsilon < epsilon_threshold:
            # return action randomly
            # print(epsilon_threshold)
            return random.choice([i for i in range(AP.C)])

        q_max: float = -sys.float_info.max
        action: int = 0
        for move in range(AP.C):
            q: float = self._predict(state, move)
            if q_max < q: 
                q_max = q
                action = move
        return action

    def update(self):
        """
        モデルの更新
        """
        for i in range(ModelParameter.epoch):
            batch_size: float = min(ModelParameter.mini_batch_size, len(self.replay_memory))
            minibatch : List[Transition] = self.replay_memory.sample(batch_size)
            target_data: List[float] = [] # 教師データ
            # build target data
            for element in minibatch:
                next_q_max: float = -sys.float_info.max
                next_action: int = 0
                # 行動選択・Q値の評価(DDQNではここが分離される)
                for next_move in range(AP.C):
                    next_q: float = self._predict(element.next_state, next_move)
                    if next_q_max < next_q:
                        next_q_max = next_q
                        # next_action = next_move
                target: float = element.reward + ModelParameter.gamma * next_q_max
                target_data.append(target)
            
            # モデルの訓練
            self.model.train()
            for element, target in zip(minibatch, target_data):
                in_ : torch.Tensor = self._build_input(element.state, element.action)
                out = self.model(in_)
                loss = self.criterion(out, torch.Tensor([target]))
                self.optimiser.zero_grad()
                loss.backward(retain_graph=True)
                self.optimiser.step()
            
    def add_memory(self, t: Transition):
        self.replay_memory.add(t)

    def build_transition(self, state: List[int], action: int, reward: int, next_state: List[int]) -> Transition:
        return Transition(state, action, reward, next_state)
    
    def get_modelname(self) -> str:
        return self.__class__.__name__
    
    def eval(self):
        self.model.eval()
    
    def display_config(self):
        print("alpha (learning rate): ", ModelParameter.alpha)
        print("memory_capacity: ", ModelParameter.memory_capacity)
        print("gamma (Q discount rate)", ModelParameter.gamma)
        print()
        print("mini_batch_size", ModelParameter.mini_batch_size)
        print("input_size: ", ModelParameter.input_size)
        print("output_size: ", ModelParameter.output_size)
        print()
        print("epsilon_start: ", ModelParameter.epsilon_start)
        print("epsilon_end: ", ModelParameter.epsilon_end)
        print("epsilon_decay: ", ModelParameter.epsilon_decay)
        print()
        print("epoch: ", ModelParameter.epoch)
