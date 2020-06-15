from typing import NamedTuple, List
import random
from collections import deque

class Transition(NamedTuple):
    state: List[int] # 2回前までにカテゴりの配列 [1step before, 2step before]
    action: int # 次に発話する発話カテゴリ番号
    reward: int # 0 or 1 or -1
    next_state: List[int] # 次の状態 [action, 1step before]

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: deque[Transition] = deque(maxlen=100)
    
    def add(self,transition: Transition):
        self.memory.append(transition)
    
    def sample(self, batch_size) -> List[Transition]:
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)