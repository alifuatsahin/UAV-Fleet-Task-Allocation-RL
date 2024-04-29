import os  
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], tau=0.005, env=None, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=256, layer2_size=256, batch_size=100):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='actor')
        self.critic1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='critic_1')
        self.critic2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target_value')

        self.update_network_parameters(tau=1)