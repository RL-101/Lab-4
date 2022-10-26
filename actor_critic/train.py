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
  hidden_units = 32
  print_every = 100
  save_fig = False
  num_steps = int(1e6)
  state_size = env.observation_space
  hidden_units = hidden_units
  num_actions = env.action_space

  # Initialize AC model
  AC_net = ActorCritic(env.observation_space, hidden_units, env.action_space)
  # Define optimizer
  optimizer = optim.Adam(AC_net.parameters(), lr=1e-3)

  episodes_passed = 0
  acc_rewards = []
  last_t = 0
  state = env.reset()

  # Initialize episodic reward list
  episodic_rewards = []
  avg_episodic_rewards = []
  stdev_episodic_rewards = []
  acc_episodic_reward = 0.0
  best_avg_episodic_reward = -np.inf
  number_episodes = 2000
  max_episode_length = 1000
  scores_deque = deque(maxlen=100)
  scores = []

  for episode in range(1,number_episodes+1):
        state = env.reset()
        rewards = []
        log_probs = []

        for step in range(max_episode_length):
            state = torch.from_numpy(state).float().to(device)
            # distribution over possible actions for state
            policy = AC_net.forward
            action_distribution, state_value = policy(state)
            # sample action fron distribution
            action = action_distribution.sample()
            # compute log probability
            log_prob = action_distribution.log_prob(action).unsqueeze(0)
            # take a step in the env
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

        total_rewards = sum(rewards)
        scores.append(total_rewards)
        scores_deque.append(total_rewards)
	#   for t in range(num_steps):
	#     if len(avg_episodic_rewards) > 0:   # so that avg_episodic_rewards won't be empty
	#         # Stop if max episodes or playing good (above avg. reward of 5 over last 10 episodes)
	#         # if episodes_passed == 5000 or avg_episodic_rewards[-1] > 5:
	#         if episodes_passed == 20000:
	#             break

	#     state = torch.from_numpy(state).float().to(device)
	#     # distribution over possible actions for state
	#     policy = AC_net.forward
	#     action_distribution, state_value = policy(state)
	#     # sample action fron distribution
	#     action = action_distribution.sample()
	#     # compute log probability
	#     log_prob = action_distribution.log_prob(action).unsqueeze(0)

	#     state, reward, done, _ = env.step(action.item())   # Get transition
	#     AC_net.rewards.append(reward)               # Document reward
	#     acc_episodic_reward = acc_episodic_reward + reward  # Document accumulated episodic reward

	#     # Episode ends - reset environment and document statistics
	#     if reward == 20:
	#     # if done:
	#         episodes_passed += 1
	#         episodic_rewards.append(acc_episodic_reward)
	#         acc_episodic_reward = 0.0

	#         # Compute average reward and variance (standard deviation)
	#         if len(episodic_rewards) <= 10:
	#             avg_episodic_rewards.append(np.mean(np.array(episodic_rewards)))
	#             if len(episodic_rewards) >= 2:
	#                 stdev_episodic_rewards.append(np.std(np.array(episodic_rewards)))

	#         else:
	#             avg_episodic_rewards.append(np.mean(np.array(episodic_rewards[-10:])))
	#             stdev_episodic_rewards.append(np.std(np.array(episodic_rewards[-10:])))

	#         # Check if average acc. reward has improved
	#         if avg_episodic_rewards[-1] > best_avg_episodic_reward:
	#             best_avg_episodic_reward = avg_episodic_rewards[-1]
	#             # if save_model:
	#             #     torch.save(AC_net, 'trained_AC_model')

	#         # Update plot of acc. rewards every 20 episodes and print
	#         # training details
	#         if episodes_passed % print_every == 0:
	#             plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
	#                          np.array(stdev_episodic_rewards), save_fig)
	#             print('Episode {}\tLast episode length: {:5d}\tAvg. Reward: {:.2f}\t'.format(
	#                 episodes_passed, t - last_t, avg_episodic_rewards[-1]))
	#             print('Best avg. episodic reward:', best_avg_episodic_reward)

	#         last_t = t  # Follow episodes length
	#         state = env.reset()
	#         AC_net.update_weights(optimizer)    # Perform network weights update
	#         continue