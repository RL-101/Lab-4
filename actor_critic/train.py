# Asynchronous Advantage Actor Critic (A3C) algorithm
from turtle import done
from model import ActorCritic
from plot import plot_rewards
import numpy as np
import matplotlib.pyplot as plt
# import pyvirtualdisplay
import gym

import torch
import torch.nn as nn

import random
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

  # Creates a virtual display for OpenAI gym
  # pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

  # set env
  env = gym.make('LunarLander-v2')
  # record video every 50 episodes
  env = gym.wrappers.Monitor(env, './videos', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

  # set seeds
  np.random.seed(0)
  torch.manual_seed(0)
  env.seed(0)

  
  # Define architecture parameters
  render = False
  gamma = 0.99
  lr = 0.02
  betas = (0.9, 0.999)
  random_seed = 543
  hidden_units = 32
  num_steps = int(1e6)
  state_size = env.observation_space
  hidden_units = hidden_units
  num_actions = env.action_space

  # Initialize AC model
  policy = ActorCritic(state_size, hidden_units, num_actions)

  # Define optimizer
  optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)

  state = env.reset()

  number_episodes = 2000
  max_episode_length = 1000
  scores = []
  score = 0

  for episode in range(1,number_episodes+1):
        state = env.reset()

        for step in range(max_episode_length):
            state = torch.from_numpy(state).float().to(device)
            # sample action from distribution
            action = policy(state)
            # take a step in the env
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            score += reward

            if done:
                break

        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.reset()
        scores.append( sum(policy.rewards))

        if score > 4000:
            break
        
        if episode % 50 == 0:
            score = score/50
            print('Episode {}\tlength: {}\t Average reward: {}'.format(episode, step, score))
            score = 0
