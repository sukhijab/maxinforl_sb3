from gymnasium import Wrapper
import numpy as np


class ActionCost(Wrapper):
    def __init__(self, env, action_cost: float = 0.0):
        assert action_cost >= 0
        super(ActionCost, self).__init__(env)
        self.action_cost = action_cost

    def step(self, action):
        observation, reward, termination, truncation, info = self.env.step(action)
        reward -= self.action_cost * np.linalg.norm(action).item()
        return observation, reward, termination, truncation, info
