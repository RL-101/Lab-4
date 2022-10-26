from turtle import done
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

import random
from collections import deque
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# some parts adapted from : https://github.com/KaleabTessera/Policy-Gradient/blob/master/reinforce/reinforce.py
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
        self.relu = F.relu
        self.softmax = F.softmax


    def forward(self, x):
        # pass observation through each operation
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        x = torch.distributions.Categorical(x)

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

    for episode in range(1,number_episodes+1):
        state = env.reset()
        rewards = []
        log_probs = []

        for step in range(1,max_episode_length+1):
            state = torch.from_numpy(state).float().to(device)
            # distribution over possible actions for state
            action_distribution = policy(state)
            # sample action fron distributio
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

        # discounted return of the trajectory
        returns = compute_returns(rewards, gamma)
        log_probs = torch.cat(log_probs)

        # sum of the product log probalities and returns (need to multiply by -1 cos we are maximizing the expected discounted return )
        policy_loss = -1 * torch.sum(log_probs * returns)

        # update the policy parameters θ
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

    returns = []
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

    for episode in range(1,number_episodes+1):
        state = env.reset()
        rewards = []
        log_probs = []

        for step in range(1,max_episode_length+1):
            state = torch.from_numpy(state).float().to(device)
            # distribution over possible actions for state
            action_distribution = policy(state)
            # sample action fron distributio
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

        # discounted return of the trajectory
        returns = compute_returns_naive_baseline(rewards, gamma)
        returns = torch.from_numpy(returns).float().to(device)
        log_probs = torch.cat(log_probs)

        # sum of the product log probalities and returns (need to multiply by -1 cos we are maximizing the expected discounted return )
        policy_loss = -1 * torch.sum(log_probs * returns)

        # update the policy parameters θ
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    return policy, scores


def run_reinforce():
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
 
    # Plot learning curve
    moving_avg = moving_average(scores, 50)
    plt.plot( scores, label='Score')
    plt.plot(
        moving_avg, label=f'Moving Average (w={50})', linestyle='--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE learning curve - CartPole-v1')
    plt.legend()
    plt.savefig('reinforce_learning_curve.png')
    plt.show()



def investigate_variance_in_reinforce():
    env = gym.make('CartPole-v1')
    seeds = np.random.randint(1000, size=5)
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)

    all_scores = []
    for seed in seeds:
        _, scores = reinforce(env=env, policy_model=policy_model, seed=int(seed), learning_rate=1e-2,
                             number_episodes=1500,
                             max_episode_length=1000,
                             gamma=1.0,
                             verbose=False)
        all_scores.append(scores)
        print(f"Reinforce training with seed:  {seed}  completed.")

    moving_avg_over_runs = np.array(
        [moving_average(score, 50) for score in all_scores])
    mean = moving_avg_over_runs.mean(axis=0)
    std = moving_avg_over_runs.std(axis=0)

    # plot varience curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(mean)+1)
    ax.plot(x, mean, '-', color='blue')
    ax.fill_between(x, mean - std, mean + std, color='blue', alpha=0.2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE averaged over 5 seeds')
    plt.savefig('reinforce_averaged_over_5_seeds.png')
    plt.show()

    return mean, std


def run_reinforce_with_naive_baseline(mean, std):
    env = gym.make('CartPole-v1')

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
    policy, scores = reinforce_naive_baseline(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    
    # Plot learning curve
    moving_avg = moving_average(scores, 50)
    plt.plot(scores, label='Score')
    plt.plot(
        moving_avg, label=f'Moving Average (w={50})', linestyle='--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE with baseline learning curve - CartPole-v1')
    plt.legend()
    plt.savefig('reinforce_with_baseline_learning_curve.png')
    plt.show()


if __name__ == '__main__':
    run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    run_reinforce_with_naive_baseline(mean, std)