from gymnasium import Wrapper


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat: int, return_total_reward: bool = False):
        assert repeat >= 1, 'Expects at least one repeat.'
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat
        self.return_total_reward = return_total_reward

    def step(self, action):
        done = False
        current_step = 0
        total_reward = 0
        while current_step < self.repeat and not done:
            observation, reward, termination, truncation, info = self.env.step(action)
            done = termination or truncation
            total_reward += reward
            current_step += 1
        reward = total_reward if self.return_total_reward else reward
        return observation, reward, termination, truncation, info