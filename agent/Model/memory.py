# Standard Imports
from typing import NamedTuple, List, Tuple, Deque, TypeVar, Generic
import random
from collections import deque

# Internal Import
from Model.sumtree import SumTree

T = TypeVar('T') # T型の定義

class SimpleMemory(Generic[T]):
    """
    batch_size分単純に保存しておくもの
    主にREINFORCEで使用する
    """
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: Deque[T] = deque(maxlen=capacity) # 古い順
    
    def add(self, transition: T):
        self.memory.append(transition)
    
    def get_all(self) -> Deque[T]:
        return self.memory
    
    def __str__(self):
        return str(self.memory)

class ReplayMemory(Generic[T]):
    """
    Experience-Replayを使用する
    """
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: Deque[T] = deque(maxlen=capacity)
    
    def add(self,transition: T):
        self.memory.append(transition)
    
    def sample(self, batch_size) -> List[T]:
        # 重複なしランダムサンプリング
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory(Generic[T]):
    """
    Prioritized Experience-Replay
    """
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory = SumTree(capacity)
        self.BIAS = 0.01
        self.a = 0.6
    
    def _get_priority(self, TDerror: float) -> float:
        return (TDerror + self.BIAS) ** self.a

    def add(self, transition: T, TDerror: float):
        p = self._get_priority(TDerror)
        self.memory.add(p, sample)
    
    def sample(self, batch_size) -> List[T]:
        batch_size = min(batch_size, len(self.memory))
        r = []
        segment = self.model.total()/batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a,b)
            (idx, p, data) = self.memory.get(s)
            r.append(data)
        return r
    
    def __len__(self):
        return len(self.memory)

    def update(self, idx, TDerror):
        p = self._getPriority(TDerror)
        self.tree.update(idx, p)
    
    def __len__(self):
        return len(self.memory)