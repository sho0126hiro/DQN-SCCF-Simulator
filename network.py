from typing import List
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from replay_memory import ReplayMemory, Transition
import constant

class DeepQNetwork:
    
    def __init__(self):
        """
        モデルの定義
        input: 2回前までの状態 (s-1, s-2)の2つと行動
        output: q値
        """
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(3,32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32,32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, 1))
        self.optimiser = optim.Adam(self.model.parameters(), 
                                    lr = constant.DeepQNetwork.LEARNING_RATE)
        # loss function
        self.criterion = nn.MSELoss()
        self.replay_memory = ReplayMemory(constant.ReplayMemory.CAPACITY)

    def build_input(self, state: List[int], action: int)-> torch.Tensor:
        """
        NNの入力データを作る
        """
        i: List[int] = []
        i.extend(state)
        i.append(action)
        return torch.Tensor(i)

    def predict(self, state: List[int], action: int) -> float:
        """
        NNを用いて推論  
        @param state: [s-1, s-2] 2つ前までの発話内容  
        @return{float} q_value Q値  
        * self.model.eval()をされている状態で関数を呼び出してください．  
        """
        self.optimiser.zero_grad() # 一度計算された勾配結果を0にリセット
        input_: torch.Tensor = self.build_input(state, action)
        q_value: float = self.model(torch.Tensor(input_))
        return q_value
    
    def replay(self, batch_size):
        """
        Experience Replayでネットワークの重みを学習する
        """
        batch_size: int = min(batch_size, len(self.replay_memory))
        minibatch: List[Transition] = random.sample(self.replay_memory.memory, batch_size)
        self.model.eval()
        target_data: List[float] = []
        # 教師データの作成
        for element in minibatch:
            next_q_list: List[float] = []
            for next_a in constant.Agent.ACTIONS:
                next_q: float = self.predict(element.next_state, next_a)
                next_q_list.append(next_q)
            next_q_max: float = np.amax(np.array(next_q_list))
            target: float = element.reward + constant.DeepQNetwork.GAMMA * next_q_max
            target_data.append(target)
        # 訓練
        self.model.train()
        for element, tartget in zip(minibatch, target_data):
            input_: torch.Tensor = self.build_input(element.state, element.action)
            out = self.model(input_)
            # 教師データとの比較
            loss = self.criterion(out, torch.Tensor(target))
            self.optimiser.zero_grad()
            loss.backward(retain_graph=True)
            self.optimiser.step()

if __name__ == "__main__":
    d = DeepQNetwork()
    d.predict([1,2],3)
