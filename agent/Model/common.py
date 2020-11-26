from Model.reinforce import REINFORCE
from Model.dqn import DQN
from Model.ddqn import DDQN
from config import AgentParameter as AP

MODEL_LIST = {
    "DQN": DQN,
    "DDQN": DDQN,
    "REINFORCE": REINFORCE
}

def get_model():
   return MODEL_LIST[AP.ALGORISM]()