# 出力されたデータを分析するクラス
import pandas as pd
import numpy as np
import csv
from config import AgentParameter as AP
from config import LoggerConfig as LC

ANALYZE_ROOT_PATH = "./analyze/log/"

class Analyzer:

    def __init__(self):
        self.out = []

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
        o_count = []
        o_rate = []
        for c, r in zip(action_count, action_rate):
            x = []
            o_count.extend(c.tolist())
            o_rate.extend(r.tolist())
        o = []
        o.extend(o_rate)
        o.extend(o_count)
        return o

if __name__ == "__main__":
    a = Analyzer()
    DIR_NAME = "20201104_011204"
    fname = LC.OUTPUT_ROOT_PATH_LOG + DIR_NAME + "/" + AP.ALGORISM
    for i in range(AP.TRY):
        df = a.read_df(fname +"_"+ str(i) + ".csv")
        a.out.append(a.ep_action_split_n(df, 3))
    tmp = np.array(a.out)
    ave = np.mean(tmp, axis=0)
    mi = np.min(tmp, axis=0)
    ma = np.max(tmp, axis=0)
    var = np.var(tmp, axis=0)
    std = np.std(tmp, axis=0)
    stde = np.std(tmp, axis=0) / np.sqrt(AP.TRY)
    med = np.median(tmp, axis=0)

    a.out.append(ave.tolist())
    a.out.append(var.tolist())
    a.out.append(std.tolist())
    a.out.append(stde.tolist())
    a.out.append(mi.tolist())
    a.out.append(ma.tolist())
    a.out.append(med.tolist())

    a.csv_write(ANALYZE_ROOT_PATH + DIR_NAME +".csv", a.out)
