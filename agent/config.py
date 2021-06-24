# Standard Import 
from typing import NamedTuple, List

class AgentParameter:
    EPISODE: int = 1500000 # episode数
    C: int = 80 # 発話内容を特徴づけるパラメータの総数
    T: int = 3 # エージェントが持つ、過去の発話履歴の長さ: T  (Tステップ前までの発話を保存)
    BATCH_SIZE: int = 10 # REINFORCEにおけるM, DQNにおけるBATCH
    REWARD: List[int] = [1, 0, -1]
    ALGORISM = "REINFORCE" # REINFORCE, DQN, DDQN
    TRY = 1 # 初期化 -> 学習　を何回繰り返すか
    
class LoggerConfig:
    OUTPUT_ROOT_PATH_IMG = "./img/"
    OUTPUT_ROOT_PATH_LOG = "./log/"
    REWARD_BATCH = 25                   