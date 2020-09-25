# Standard Import
from typing import NamedTuple, List
from datetime import datetime

# ThirdParty Import
import matplotlib.pyplot as plt
import pandas as pd

# Internal Import
from config import AgentParameter as AP
from config import LoggerConfig as LC

class BaseLogger:
    TIME_FORMAT = "%Y%m%d_%H%M%S"
    def get_now_time_str(self):
        return datetime.now().strftime(self.TIME_FORMAT)

class BehaviorHistory(NamedTuple):
    state: List[int]
    aciton: int
    reward: int

class BeheviorLogger(BaseLogger):
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
        for idx, e in enumerate(self.history):
            tmp += e.reward
            if idx % LC.REWARD_BATCH == 0:
                rh.append(tmp)
                tmp = 0
        return rh
    
    def _build_history(self):
        history = []
        for idx, e in enumerate(self.history):
            tmp = []
            for s in e.state:
                tmp.append(s)
            tmp.append(e.aciton)
            tmp.append(e.reward)
            history.append(tmp)
        return history       

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
        fig = plt.figure(1)
        rh: List[int] = self._build_reward_history()
        plt.plot(rh)
        extension = ".png"
        modelname: str = agent.model.get_modelname()
        fname: str = self._build_file_name(
            self.get_now_time_str(), modelname, "C", AP.C , "T", AP.T
        ) + extension
        fig.savefig(LC.OUTPUT_ROOT_PATH_IMG + fname)
    
    def _get_settings(self, agent):
        return [
            "model: " + self.agent.modelname,
            "episode: " + str(AP.TRY),
            "T: "+ str(AP.T),
            "Num of Category: "+ str(AP.C),
            "M(Batch): "+ str(AP.BATCH_SIZE),
        ]

    def save_csv(self, agent):
        h: List[int] = self._build_history()
        s = self._get_settings(agent)
        df = pd.DataFrame(h)
        fname = self._build_file_name(
            self.get_now_time_str(),
            agent.model.get_modelname()
        )
        df.to_csv(LC.OUTPUT_ROOT_PATH_LOG + fname + ".csv")

if __name__ == "__main__":
    pass
