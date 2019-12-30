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
