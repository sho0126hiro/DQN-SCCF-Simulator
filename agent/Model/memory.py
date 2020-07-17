# Standard Imports
from typing import NamedTuple, List, Tuple, Deque, TypeVar, Generic
import random
from collections import deque

T = TypeVar('T') # T型の定義

class SimpleMemory(Generic[T]):
    """
    batch_size分単純に保存しておくもの
    主にREINFORCEで使用する
    """
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: deque[T] = deque(maxlen=capacity) # 古い順
    
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
        self.memory: deque[T] = deque(maxlen=capacity)
    
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
    Prioritized Experience-Replayを使用する
    """
    pass