# Standard Import 
from abc import ABCMeta, abstractmethod

class AbstractModel(metaclass=ABCMeta):
    """
    Modelの Abstract Class（基底クラス）
    Modelは全てこれを継承し、abstractmethodを実装しなければならない
    """
    @abstractmethod
    def get_action(self, state):
        """
        モデルから予測して、現在の状態で最適行動を出力する
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        モデルの更新を行う
        （PolicyTableやNNの更新など）
        """
        pass

    @abstractmethod
    def add_memory(self, t): 
        """
        メモリに書き込む
        """   
        pass

    @abstractmethod
    def build_transition(self, state, action, reward, next_state):
        """
        メモリに格納するTransitionObjectを作成する
        """
        pass
