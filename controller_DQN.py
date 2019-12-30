from collections import namedtuple
from itertools import count
import torch.nn.functional as F

import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch.autograd import Variable
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear((32+3*256), 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.predictor(x)
        return torch.softmax(x, dim=1)


class Controller(nn.Module):

    def __init__(self, gamma):
        super(Controller, self).__init__()

        # 32 latent vars from VAE, 100 hidden state vars from LSTM
        self.predictor = nn.Sequential(
            nn.Linear((32+3*256), 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.gamma = gamma

        self.policy_history = Variable(torch.Tensor()).cuda()
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = self.predictor(x) / 1

        return torch.softmax(x, dim=1)


def select_action(state, policy):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state

    # print(state)
    preds = policy(state)
    print(preds)
    c = Categorical(preds)
    action = c.sample()

    # Add log probability of our chosen action to our history
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history.to(device), c.log_prob(action).to(device)])
    else:
        policy.policy_history = (c.log_prob(action).to(device))
    return action

## TODO create replay buffer



def update_policy(policy, optimizer):
    R = 0
    rewards = []

    # Reward discounting (i.e. smear reward back through history)
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards, just some autoscaling
    rewards = torch.FloatTensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps).to(device)
    # print(rewards)
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history.to(device), Variable(rewards)).mul(-1).to(device), -1)).to(device)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode).item())
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []
