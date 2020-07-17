# Standard Import
from typing import NamedTuple, List

# ThirdParty Import
import matplotlib.pyplot as plt
import pandas as pd

# Internal Import
from config import AgentParameter as AP
from config import LoggerConfig as LC

class BehaviorHistory(NamedTuple):
    state: List[int]
    aciton: int
    reward: int

class BeheviorLogger:
    """
    データの描画・書き出し用クラス
    """
    def __init__(self):
        self.history: List[BehaviorHistory] = []

    def add(self, state: List[int], action: int, reward: int):
        self.history.append(
            BehaviorHistory(state, action, reward)
        )
    
    def _build_reward_history(self) -> List[int]:
        rh: List[int] = []
        tmp = 0
        print(len(self.history))
        for idx, e in enumerate(self.history):
            tmp += e.reward
            if idx % LC.REWARD_BATCH == 0:
                rh.append(tmp)
                tmp = 0
        return rh
    
    def _build_file_name(self, *args) -> str:
        """
        引数（str)を`_`区切りで結合する
        """
        s: str = ""
        for e in args:
            s += "_" + str(e)
        return s.strip("_")
    
    def save_reward_graph(self, agent):
        """
        獲得報酬の推移グラフを出力する
        """
        # print(agent.model.get_modelname())
        rh: List[int] = self._build_reward_history()
        plt.plot(rh)
        extension = ".png"
        modelname: str = agent.model.get_modelname()
        fname: str = self._build_file_name(
            modelname, "C", AP.C , "T", AP.T
        ) + extension
        print(fname)
        plt.savefig(LC.OUTPUT_ROOT_PATH + fname)

if __name__ == "__main__":
    pass
