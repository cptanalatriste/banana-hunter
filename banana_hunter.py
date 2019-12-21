from collections import namedtuple, deque
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cude:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    """
    A repository of agent experiences. While training, experiences will be
    provided from here.
    """

    def __init__(self, num_actions, buffer_size, training_batch_size):

        self.num_actions = num_actions
        self.storage = deque(maxlen=buffer_size)
        self.training_batch_size = training_batch_size

        self.field_names = ['state', 'action', 'reward', 'next_state', 'done']
        self.field_types = [np.float32, np.int64, np.float32, np.float32, np.float32]

        self.experience = namedtuple("Experience", field_names=self.field_names)

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the repository.
        """
        self.storage.append(self.experience(state, action, reward,
                                            next_state, done))

    def sample(self):
        """
        Sample a set of experiences from the repository.
        """
        raw_samples = random.sample(self.storage, k=self.training_batch_size)
        values = []

        for tuple_index in range(len(self.field_names)):
            value_list = [sample[tuple_index]
                          for sample in raw_samples if sample is not None]

            value_list = np.vstack(value_list).astype(self.field_types[tuple_index])
            value_list = torch.from_numpy(value_list)
            values.append(value_list.to(DEVICE))

        return tuple(values)

    def __len__(self):
        return len(self.storage)

class BananaAgent():
    """
    A deep-reinforcement learning agent hungry for bananas.
    """

    def __init__(self, state_size, num_actions, buffer_size=int(1e5),
                 learning_frequency=4, min_learning_samples=64, gamma=0.99,
                 second_layer_input=64, second_layer_output=64,
                 learning_rate=5e-4, tau=1e-3):

        self.state_size = state_size
        self.num_actions = num_actions
        self.min_learning_samples = min_learning_samples
        self.target_network = self.get_q_network(second_layer_input,
                                                 second_layer_output)
        self.local_network = self.get_q_network(second_layer_input,
                                                second_layer_output)

        self.replay_buffer = ReplayBuffer(num_actions=num_actions,
                                          buffer_size=buffer_size,
                                          training_batch_size=min_learning_samples)
        self.optimizer = optim.Adam(self.local_network.parameters(),
                                    lr=learning_rate)

        self.step_counter = 0
        self.gamma = gamma
        self.tau = tau
        self.learning_frequency = learning_frequency

    def get_q_network(self, second_layer_input, second_layer_output):
        """
        This method defines the neural network architecture of our agent.
        """
        model = nn.Sequential(
            nn.Linear(in_features=self.state_size, out_features=second_layer_input),
            nn.ReLU(),
            nn.Linear(in_features=second_layer_input, out_features=second_layer_output),
            nn.ReLU(),
            nn.Linear(in_features=second_layer_output, out_features=self.num_actions))

        return model

    def step(self, state, action, reward, next_state, done):
        """
        Receives information from the environment. Is in charge of storing
        experiences in the replay buffer and triggering agent learning.
        """

        self.replay_buffer.add(state, action, reward, next_state, done)
        self.step_counter += 1

        if (self.step_counter % self.learning_frequency and
                len(self.replay_buffer) > self.min_learning_samples):

            learning_samples = self.replay_buffer.sample()
            self.learn(learning_samples)

    def learn(self, learning_samples):
        """
        A lerning step for our agent. It consists in obtaining q-values from
        the local network, comparing them with what's produced from the
        target network, and update accordingly.
        """

        states, actions, rewards, next_states, dones = learning_samples

        target_next_states = self.target_network(next_states).detach()
        target_qvalues_max, _ = target_next_states.max(dim=1)
        target_qvalues_max = target_qvalues_max.unsqueeze(1)

        target_q_values = rewards + (self.gamma * target_qvalues_max * (1 - dones))

        local_current_states = self.local_network(states)
        local_values_per_action = local_current_states.gather(dim=1, index=actions)

        loss = F.mse_loss(input=local_values_per_action, target=target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_model_parameters()

    def update_model_parameters(self):
        """
        Update the weights of the target network with information gathered from
        the local network.
        """

        for target_parameter, local_parameter in zip(self.target_network.parameters(),
                                                     self.local_network.parameters()):

            local_value = local_parameter.data
            target_value = target_parameter.data

            update_value = self.tau * local_value + (1.0 - self.tau) * target_value
            target_value.copy_(update_value)

    def save_trained_weights(self, network_file):
        """
        Stores the weights of the local network in a file.
        """

        torch.save(self.local_network.state_dict(), network_file)
        print("Network state saved at ", network_file)

    def load_trained_weights(self, network_file):
        """
        Takes weights from a file and assigns them to the local network.
        """

        self.local_network.load_state_dict(torch.load(network_file))
        print("Network state loaded from ", network_file)

    def act(self, state, epsilon=0.0):
        """
        Given a state, produces an action to perform. Epsilon, if provided,
        is the probability of performing an action at random.
        """
        action = None
        if random.random() > epsilon:

            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            self.local_network.eval()
            with torch.no_grad():
                qvalues_per_action = self.local_network(state)
            self.local_network.train()

            action = np.argmax(qvalues_per_action.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.num_actions))

        return action
