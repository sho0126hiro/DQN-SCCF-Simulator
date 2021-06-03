# Standard Import
import os
import sys
import random
import pprint
from typing import List

# ThirdParty Import
import numpy as np

# Internal Import
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Evaluator.evaluator import AbstractEvaluator
from config import AgentParameter as AP

class EvaluatorMemory:
    def __init__(self, init_state: List[int]):
        reverse_state = init_state[::-1]
        tmp = reverse_state[0]
        count = 0
        for e in reverse_state:
            if tmp != e:
                break
            count += 1
        self.t: int = count # 同じ行動を繰り返した回数
        self.a: int = init_state[-1] # 前回の行動番号

    def update(self, action: int):
        if action == self.a:
            self.t += 1
        else:
            self.t = 0
        self.a = action

class HumanizedEvaluator(AbstractEvaluator):
    """
    人間を模倣した評価器
    """
    def __init__(self, init_state: int):
        """
        @param init_state: メモリ初期化（直前の行動予測）用
        """
        self.prob = ExTestUser01.prob
        self._prob_init()
        self.alpha: float = 0.8 # 減衰係数
        self.bias: float = 0.001 # 0％の箇所にいれるバイアス
        self.memory = EvaluatorMemory(init_state) # データ保存先

    def _prob_init(self):
        """
        self.probを初期化する（C）に合わせる
        """
        if AP.C == 6:
            return
        # idx_list: List[int] = random.choices([i for i in range(6)], k=AP.C)
        idx_list: List[int] =[i % 6 for i in range(AP.C)]
        p = []
        for r in self.prob:
            tmp = []
            for idx in idx_list:
                tmp.append(r[idx])
            p.append(tmp)
        self.prob = p
        print(idx_list)

    def _roulette(self, list_: List[float]) -> int:
        """
        ルーレット選択をする
        """
        prob:List[float] = (np.array(list_) / sum(list_)).tolist()
        return np.random.choice(AP.REWARD, p = prob)

    def evaluate(self, state: List[int], action: int) -> int:
        self.memory.update(action)
        p: List[float] = []
        for r in range(len(AP.REWARD)):
            if len(self.prob[r][action]) > self.memory.t: 
                if self.prob[r][action][self.memory.t] == 0:  
                    self.prob[r][action][self.memory.t]+=self.bias
                p.append(self.prob[r][action][self.memory.t])
            else:
                if self.prob[r][action][-1] == 0:  
                    self.prob[r][action][-1]+=self.bias
                t: int = self.memory.t - len(self.prob[r][action])+1
                alpha: float = self.alpha ** t
                r1: float = self.prob[r][action][-1] * alpha
                s: float = self.prob[r+1][action][-1] + self.prob[r+2][action][-1]
                r0: float = (1 - r1) * self.prob[r+1][action][-1] / s
                rm1: float = (1 - r1) * self.prob[r+2][action][-1] / s
                p.append(r1)
                p.append(r0)
                p.append(rm1)
                break
        r = self._roulette(p)
        return r

class User01:
    prob = [
        [ # r = 1
            [0.53, 0.91, 0.92, 1.0, 1.0, 1.0, 1.0],
            [0.4, 0.44],
            [0.48, 0.13],
            [0.35, 0.5],
            [0.45, 0.0, 0.0],
            [0.38, 0.0, 0.0],
        ],
        [ # r = 0
            [0.46,0.09,0.08, 0.0, 0.0, 0.0, 0.0],
            [0.55,0.33],
            [0.50,0.88],
            [0.54,0.50],
            [0.39,1.00,1.00],
            [0.50,1.00,1.00],
        ],
        [ # r = -1
            [0.01, 0.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0], 
            [0.05, 0.22],
            [0.02, 0.0],
            [0.11, 0.0],
            [0.16, 0.0, 0.0], 
            [0.12, 0.0, 0.0], 
        ]
    ]

class User04:
    prob = [
        [ # r = 1
            [0.37, 0.78, 0.75, 1.0],
            [0.35, 0,29, 0],
            [0.44, 0.29],
            [0.35, 0.5],
            [0.36, 0.17, 0],
            [0.24, 0],
            [0.21, 0]
        ],
        [ # r = 0
            [0.61, 0.22, 0.25, 0],
            [0.58, 0.43, 1],
            [0.48, 0.43],
            [0.64, 0.83,1],
            [0.67, 1],
            [0.74, 1]
        ],
        [ # r = -1
            [0.02, 0, 0, 0],
            [0.06, 0.28, 0],
            [0.08, 0.28],
            [0, 0, 0],
            [0.08, 0],
            [0.04, 0]
        ]
    ]

class User07:
    prob = [
        [ # r = 1
            [0.52, 0.75],
            [0.51, 0.77, 0.67], 
            [0.45, 0.50, 0.00],
            [0.51, 0.53, 1.00],
            [0.48, 0.00],
            [0.56, 0.00, 0.00], 
        ],
        [ # r = 0
            [0.48, 0.25],
            [0.48, 0.23, 0.33], 
            [0.55, 0.50, 1.00],
            [0.46, 0.47, 0.00],
            [0.52, 1.00],
            [0.44, 1.00, 1.00]
        ],
        [ # r = -1
            [0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.03, 0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00, 0.00]
        ]
    ]

class User08:
     prob = [
        [ # r = 1
            [0.28, 0.50],
            [0.32, 0.22, 0.33],
            [0.23, 0.27],
            [0.16, 0.33, 0.33],
            [0.30, 0.00, 0.00, 0.00],
            [0.24, 0.00]
        ],
        [ # r = 0
            [0.47, 0.43],
            [0.34, 0.56, 0.33, 0.00],
            [0.50, 0.73],
            [0.43, 0.00, 0.00],
            [0.43, 0.00, 0.00],
            [0.41, 0.14]
        ],
        [ # r = -1
            [0.25, 0.07],
            [0.34, 0.22, 0.33],	
            [0.27, 0.00],
            [0.41, 0.67, 0.67],
            [0.27, 1.00 , 1.00, 1.00],
            [0.35, 0.86]
        ]
    ]


class ExTestUser01: # 4,5,6 が特に評価が高いエージェント
    prob =[
        [ # r = 1
            [0.1],
            [0.1],
            [0.3],
            [0.6],
            [0.7],
            [0.7],
        ],
        [ # r = 0
            [0.2],
            [0.7],
            [0.4],
            [0.2],
            [0.05],
            [0.1],
        ],
        [ # r = -1
            [0.7],
            [0.2],
            [0.3],
            [0.2],
            [0.25],
            [0.2],
        ]
    ]


if __name__ == "__main__":
    e = HumanizedEvaluator()
    for _ in range(10):
        e.evaluate(None, 3)
