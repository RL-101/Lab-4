from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from gym import spaces
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    A basic implementation of a Actor critic Network. 
    """

    def __init__(self, state_size: spaces.Box, hidden_units, num_actions: spaces.Discrete):
        """
        Initialise the Actor and critic
        :var fc1_dims: the number of dimensions for the first fully connected layer
        :var fc2_dims: the number of dimensions for the second fully connected layer
        :param n_actions: number of actions
        :param state_size: obersvation space
        :param hidden_size: size of neurons in the hidden layers
        """

        super().__init__()
        
        self.fc1_dims = hidden_units
        self.fc2_dims = hidden_units
        self.num_actions = num_actions.n
        self.input = nn.Linear(4, 128)
        self.state_size = state_size.shape[0]
        self.saved_actions = []
        self.rewards = []

        # fully connected layers
        self.fc1 = nn.Linear(self.state_size, self.fc1_dims)

        # output layers
        self.actor = nn.Linear(self.fc2_dims, self.num_actions)
        self.critic = nn.Linear(self.fc2_dims, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))

        state_values = self.critic(x)

        actor = self.actor(x).unsqueeze(0)
        action_prob = F.softmax(actor, dim=1)
        distribution = Categorical(action_prob)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return distribution, state_values

