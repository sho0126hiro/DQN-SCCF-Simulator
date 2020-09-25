# Standard Import 
from typing import NamedTuple, List

class AgentParameter:
    EPISODE: int = 300 # episode数
    C: int = 20 # 発話内容を特徴づけるパラメータの総数
    T: int = 2 # エージェントが持つ、過去の発話履歴の長さ: T  (Tステップ前までの発話を保存)
    BATCH_SIZE: int = 3 # REINFORCEにおけるM, DQNにおけるBATCH
    REWARD: List[int] = [1, 0, -1]

class LoggerConfig:
    OUTPUT_ROOT_PATH_IMG = "./img/"
    OUTPUT_ROOT_PATH_LOG = "./log/"
    REWARD_BATCH = 25