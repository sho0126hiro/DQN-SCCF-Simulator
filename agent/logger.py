# Standard Import
from typing import NamedTuple, List
from datetime import datetime
import os

# ThirdParty Import
import matplotlib.pyplot as plt
import pandas as pd

# Internal Import
from config import AgentParameter as AP
from config import LoggerConfig as LC
from analyzer import Analyzer

class BaseLogger:
    TIME_FORMAT = "%Y%m%d_%H%M%S"
    first_try = True

    def get_now_time_str(self):
        if(self.first_try):
            self.t = datetime.now().strftime(self.TIME_FORMAT)
            os.mkdir(LC.OUTPUT_ROOT_PATH_LOG + self.t )
            self.first_try = False
        return self.t

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
            "model: " + agent.modelname,
            "episode: " + str(AP.EPISODE),
            "T: "+ str(AP.T),
            "Num of Category: "+ str(AP.C),
            "M(Batch): "+ str(AP.BATCH_SIZE)
        ]

    def _make_column_name(self, h, s):
        ret = []
        for i in range(AP.T, 0, -1):
            ret.append("s-" + str(i))
        ret.append("a")
        ret.append("r")
        if len(h[0]) < len(s):
            for _ in range(len(s) - len(h[0])):
                ret.append("none")
        return ret
    
    def save_csv(self, agent, index):
        h: List[int] = self._build_history()
        s = self._get_settings(agent)
        h.append(s)
        df = pd.DataFrame(h, columns=self._make_column_name(h,s))
        fname = self._build_file_name(
            agent.model.get_modelname(),
            index
        )
        output_filename = LC.OUTPUT_ROOT_PATH_LOG + self.get_now_time_str() + "/" + fname + ".csv"
        print(output_filename)
        df.to_csv(output_filename)
        return output_filename
    
    def reset(self):
        self.history: List[BehaviorHistory] = []

if __name__ == "__main__":
    pass
