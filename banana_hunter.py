import numpy as np
from collections import namedtuple, deque
import random

class ReplayBuffer():

    def __init__(self, num_actions, buffer_size, training_batch_size):

        self.num_actions = num_actions
        self.storage = deque(maxlen=buffer_size)
        self.training_batch_size = training_batch_size

        self.experience = namedtuple("Experience",
            field_names=['state', 'action', 'reward', 'next_state', 'done'])


    def add(self, state, action, reward, next_state, done):
        self.storage.append(self.experience(state, action, reward,
            next_state, done))

    def sample(self):
        raw_samples = random.sample(self.storage, k = self.training_batch_size)

    def __len__(self):
        return len(self.storage)

class BananaAgent():

    def __init__(self, state_size, num_actions, buffer_size=int(1e5), learning_frequency=4,
        min_learning_samples=64):

        self.state_size = state_size
        self.num_actions = num_actions
        self.min_learning_samples = min_learning_samples

        self.replay_buffer = ReplayBuffer(num_actions=num_actions,
            buffer_size= buffer_size,
            training_batch_size=min_learning_samples)

        self.step_counter = 0
        self.learning_frequency = learning_frequency


    def step(self, state, action, reward, next_state, done):

        self.replay_buffer.add(state, action, reward, next_state, done)
        self.step_counter += 1

        if (self.step_counter % self.learning_frequency and
            len(self.replay_buffer) > self.min_learning_samples):

            learning_samples = self.replay_buffer.sample()

    def act(self, state, epsilon=None):
        action = np.random.randint(self.num_actions)
        return action
