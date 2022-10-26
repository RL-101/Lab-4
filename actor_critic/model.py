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
        
        self.logprobs = []
        self.rewards = []
        self.state_values  = []

        # fully connected layers
        self.fc1 = nn.Linear(self.state_size, self.fc1_dims)

        # output layers
        self.actor = nn.Linear(self.fc2_dims, self.num_actions)
        self.critic = nn.Linear(self.fc2_dims, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # the value from state s_t
        state_values = self.critic(x)

        actor = self.actor(x).unsqueeze(0)
        action_prob = F.softmax(actor, dim=1)
        # a list with the probability of each action over the action space
        action_distribution = Categorical(action_prob)


        # select action
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_values)
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss

    def reset(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

