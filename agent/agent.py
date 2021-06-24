# Standard Import
from typing import List, Deque
from collections import deque

# ThirdParty Import
import numpy as np

# Internal Import
from Model.reinforce import REINFORCE
from Model.dqn import DQN
from Model.ddqn import DDQN
from Model.common import get_model
from Evaluator.random_evaluator import RandomEvaluator
from Evaluator.humanized_evaluator import HumanizedEvaluator
from config import AgentParameter as AP
from logger import BeheviorLogger
from analyzer import Analyzer

class Agent:
    
    def display_config(self):
        print("---------- Agent Config ----------")
        print("modelname: ", self.modelname)
        print("C: ", AP.C)
        print("T: ", AP.T)
        print("Batch_size: ", AP.BATCH_SIZE)
        print("rewards: ", AP.REWARD)
        
        print("---------- Model Config ----------")
        self.model.display_config()

    def __init__(self, logger):
        self.model = get_model()
        self.modelname = self.model.get_modelname()
        self.state: Deque = deque(maxlen=AP.T) # [t回前のc, t-1回目のc, t-2..., 1回前のc]
        self._init_state()
        self.evaluator = HumanizedEvaluator(list(self.state))
        self.logger = logger
        self.analyzer = Analyzer()
        self.display_config()
    
    def _init_state(self):
        for i in range(AP.T):
            self.state.append(np.random.choice(AP.C))

    def _update_state(self,action):
        self.state.append(action)

    def run(self, index=0):
        c = 0
        for t in range(int(AP.EPISODE/AP.BATCH_SIZE)):
            c+=1
            # print(c)
            for episode in range(AP.BATCH_SIZE):
                if self.modelname != "REINFORCE": self.model.eval()
                s: List[int] = list(self.state)
                action: int = self.model.get_action(self.state)
                reward: int = self.evaluator.evaluate(s, action) # reward
                self._update_state(action)
                self.model.add_memory(
                    self.model.build_transition(
                        s, action, reward, list(self.state)
                    )
                )
                self.logger.add(s, action, reward)
            self.model.update()
        output_filename = self.logger.save_csv(self, index)
        print("output_filename: ", output_filename)
        result_dataframe = self.analyzer.read_df(output_filename)
        self.analyzer.ep_action_split_n(result_dataframe, 3)
        self.logger.reset()

def main():
    logger = BeheviorLogger()
    for i in range(AP.TRY):
        a = Agent(logger)
        a.run(i)

if __name__ == "__main__":
    main()
    