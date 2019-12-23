import numpy as np
import torch.nn as nn
import torch
import socket
from PIL import Image
from torch.autograd import Variable
from torch.distributions import Categorical

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Agent:
    def __init__(self):
        pass

    def step(self, end, reward, state):
        # return 0 # nothing
        # return 1 # left
        # return 2 # right
        return 3 # random


learning_rate = 0.01
gamma = 0.99

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.policymodel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5,dilation=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, dilation=7),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, dilation=7),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )

        self.gamma = gamma

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []


    def forward(self, x):
        x = torch.FloatTensor(x).reshape(1,64,64, 3)
        x = self.policymodel(x)  # Made the softmax temp lower
        print(x)
        return torch.softmax(x, dim=1)



class Environment:
    def __init__(self, ip = "127.0.0.1", port = 13000, size = 768, timescale = 1):
        self.client     = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip         = ip
        self.port       = port
        self.size       = size
        self.timescale  = timescale

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def state2image(self, state):
        return d

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data   = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end    = data[0]
        reward = data[1]
        state  = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))


#data   = self.client.recv(1 + 1 + 768 * 768 * 3, socket.MSG_WAITALL)
