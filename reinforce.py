from turtle import done
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

import random
from collections import deque
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    '''
        approximate policy using a simple Neural Network
        params : 
                observation provide by the environment
        return :
                action to take
    '''
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(SimplePolicy, self).__init__()

        # Input to hidden layer linear transformation
        self.hidden = nn.Linear(s_size, h_size)
        # outputlayer 
        self.output = nn.Linear(h_size, a_size)
        # relu activation and softmax output
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # pass observation through each operation
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x





def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    '''
        params : 
                rewards - an array of rewards
                gamma -  discount factor
        returns :
                return of a trajectory
        G_t = R_t_1  + gamma*R_t_2 + (gamma^2)*R_t_3 ...
    '''
    returns = 0
    for i in range(len(rewards)):
        returns += pow(gamma,i) * rewards[i]

    return returns


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.seed(seed)

    policy = policy_model.to(device)
    scores = []
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    scores_deque = deque(maxlen=100)

    for episode in range(len(number_episodes)):
        state = env.reset()
        rewards = []
        log_probs = []

        for step in range(max_episode_length):
            
            action,log_prob = policy.act(state)
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

    scores.append(sum(rewards))
    scores_deque.append(sum(rewards))
    returns = compute_returns(rewards, gamma)
    log_probs = torch.cat(log_probs)
    policy_loss = -torch.sum(log_probs * returns)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


    return policy, scores


def compute_returns_naive_baseline(rewards, gamma):
    '''
        params :
                rewards - an array of rewards
                gamma - discount factor γ
        return:
                return of a trajectory at each time-step.
        
        take average of the rewards over a single trajectory
        and normalise the rewards by dividing by the standard deviation
    '''

    returns = 0
    r = 0
    for i in reversed(range(len(rewards))):
        r = gamma * r + rewards[i]
        returns.insert(0, r)
    
    returns = np.array(returns)
    mean= returns.mean()
    std = returns.std(axis=0)
    returns = (returns - mean)/std
    return returns


def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    policy = policy_model.to(device)
    scores = []
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    scores_deque = deque(maxlen=100)

    for episode in range(len(number_episodes)):
        state = env.reset()
        rewards = []
        log_probs = []

        for step in range(max_episode_length):
            
            action,log_prob = policy.act(state)
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

    scores.append(sum(rewards))
    scores_deque.append(sum(rewards))
    returns = compute_returns(rewards, gamma)
    log_probs = torch.cat(log_probs)
    policy_loss = -torch.sum(log_probs * returns)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy, scores
    return policy, scores

def run_reinforce():
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    # Plot learning   curve


def investigate_variance_in_reinforce():
    env = gym.make('CartPole-v1')
    seeds = np.random.randint(1000, size=5)

    raise NotImplementedError

    return mean, std
  

def run_reinforce_with_naive_baseline(mean, std):
    env = gym.make('CartPole-v1')

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    raise NotImplementedError


if __name__ == '__main__':
    run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    run_reinforce_with_naive_baseline(mean, std)
