from Model.reinforce import REINFORCE
from Model.dqn import DQN
from Model.dqn_stochastically import DQN_STOC as DQN2
from Model.ddqn import DDQN
from config import AgentParameter as AP

MODEL_LIST = {
    "DQN": DQN,
    "DDQN": DDQN,
    "REINFORCE": REINFORCE,
    "DQN_STOC": DQN2,
}

def get_model():
   return MODEL_LIST[AP.ALGORISM]()