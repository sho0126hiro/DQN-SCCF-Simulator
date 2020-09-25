# 出力されたデータを分析するクラス
import pandas as pd
import numpy as np
import csv
from config import AgentParameter as AP
from config import LoggerConfig as LC

class Analyzer:

    def read_df(self, filename):
        """
        データフレーム取得（csvからimport)
        データフレームの形式: 

        """
        df = pd.read_csv(filename)
        df.drop(df.tail(1).index, inplace=True)
        return df
    
    def csv_write(self, fname, writedata):
        with open(fname,'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(writedata)
        
        
    def ep_action_split_n(self, df, n: int):
        """
        エピソードをn分割し、それぞれの発話確率を示す。
        """
        action_count = np.zeros((n,AP.C))
        action_rate = np.zeros((n,AP.C))
        count = np.zeros(n)
        split_count = np.zeros(AP.C)
        for index, action in enumerate(df["a"]):
            action = int(action)
            for j in range(n):
                if (j) * len(df) / n <= index < (j+1) * len(df) / n:
                    action_count[j][action] += 1
                    count[j] += 1
        for index, action in enumerate(action_count):
            action_rate[index] = action/count[index]
        
        action_count.tolist()
        action_rate.tolist()
        out = []
        for c, r in zip(action_count, action_rate):
            out.append(c.tolist())
            out.append(r.tolist())
        self.csv_write("out.csv", out)

if __name__ == "__main__":
    a = Analyzer()
    fname = LC.OUTPUT_ROOT_PATH_LOG + "20200925_234451_REINFORCE.csv"
    df = a.read_df(fname)
    a.ep_action_split_n(df, 3)

