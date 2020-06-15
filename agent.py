import numpy as np
from typing import List
from network import DeepQNetwork
import constant
from evaluator import ProbEvaluator
from replay_memory import Transition
import matplotlib.pyplot as plt

class Agent:
    def __init__(self):
        self.dqn = DeepQNetwork()
        self.epsilon = 1.0
        self.evaluator = ProbEvaluator()

    def select_next_action(self, state: List[int]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(constant.Agent.ACTIONS)
        else:
            return self.select_best_action(state)
    
    def select_best_action(self, state) -> int:
        actions = constant.Agent.ACTIONS
        q_max: float = -float('inf')
        for a in actions:
            q: float = self.dqn.predict(state, a)
            # print(a,q)
            if q_max < q:
                q_max = q
                next_ = a
        return next_
    
    def init_state(self) -> List[int]:
        s1 = np.random.choice(constant.Agent.ACTIONS)
        s2 = np.random.choice(constant.Agent.ACTIONS)
        return [s1,s2]
    
    def update_epsilon(self):
        if self.epsilon > constant.Agent.EPSILON_MIN:
            self.epsilon *= constant.Agent.EPSILON_DECAY

    def run(self):
        state = self.init_state()
        ave_reward_list = []
        for _ in range(constant.Agent.TRY):
            ave_reward = 0
            for episode in range(constant.Agent.EPISODE):
                action: int = self.select_next_action(state)
                reward: int = self.evaluator.evaluate(state, action)
                next_state: List[int] = [action, state[0]]
                # print(state, action, next_state , reward)
                transition: Transition = Transition(state, action, reward, next_state)
                self.dqn.replay_memory.add(transition)
                state = next_state
                ave_reward += reward
            ave_reward_list.append(ave_reward)
            self.dqn.replay(32)
            # print(self.epsilon)
            self.update_epsilon()
            if _ % 100 == 0:
                print(_)
               
        # check prob
        counter = [0 for i in range(constant.Agent.C)]
        for _ in range(constant.Agent.CHECK):
            action = self.select_best_action(state)
            counter[action] += 1
            next_state: List[int] = [action, state[0]]
        print(counter)

        plt.plot(ave_reward_list)
        plt.show()

if __name__ == "__main__":
    a = Agent()
    a.run()