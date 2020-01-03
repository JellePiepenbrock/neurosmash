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
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DQN2(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.r2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.r3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.flat = Flatten()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # print(linear_input_size)
        linear_input_size = 800
        # self.l1 = nn.Linear(linear_input_size, 64)
        # self.r4 = nn.ReLU()
        # self.l2 = nn.Linear(linear_input_size, 256)
        # self.r5 = nn.ReLU()
        self.out = nn.Linear(linear_input_size, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.bn1(self.r1(self.conv1(x)))
        x = self.bn2(self.r2(self.conv2(x)))
        x = self.bn3(self.r3(self.conv3(x)))
        
        x = self.flat(x)
        
        # x = self.r4(self.l1(x))
        # x = self.r5(self.l2(x))
        x = self.out(x)
        return x
