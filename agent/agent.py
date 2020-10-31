# Standard Import
from typing import List, Deque
from collections import deque

# ThirdParty Import
import numpy as np

# Internal Import
from Model.reinforce import REINFORCE
from Model.dqn import DQN
from Model.ddqn import DDQN
from Evaluator.random_evaluator import RandomEvaluator
from Evaluator.humanized_evaluator import HumanizedEvaluator
from config import AgentParameter as AP
from logger import BeheviorLogger
from analyzer import Analyzer

class Agent:
    
    def __init__(self):
        # self.model = REINFORCE()
        self.model = DQN()
        # self.model = DDQN()
        self.modelname = self.model.get_modelname()
        self.state: Deque = deque(maxlen=AP.T) # [t回前のc, t-1回目のc, t-2..., 1回前のc]
        self._init_state()
        self.evaluator = HumanizedEvaluator(list(self.state))
        self.logger = BeheviorLogger()
        self.analyzer = Analyzer()
    
    def _init_state(self):
        for i in range(AP.T):
            self.state.append(np.random.choice(AP.C))

    def _update_state(self,action):
        self.state.append(action)

    def run(self):
        for t in range(int(AP.EPISODE/AP.BATCH_SIZE)):
            print(t)
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
        # self.logger.save_reward_grxaph(self)
        output_filename = self.logger.save_csv(self)
        result_dataframe = self.analyzer.read_df(output_filename)
        self.analyzer.ep_action_split_n(result_dataframe, 3)

if __name__ == "__main__":
    a = Agent()
    a.run()
