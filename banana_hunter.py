import numpy as np


class BananaAgent():

    def __init__(self, state_size, num_actions):
        self.state_size = state_size
        self.num_actions = num_actions

    def act(self, state, epsilon=None):
        action = np.random.randint(self.num_actions)
        return action

    def step(self, state, action, reward, next_state, done):
        return None
