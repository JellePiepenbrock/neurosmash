from VAE import VAE
from rnn_vae import MDNRNN
from controller import Controller, select_action, update_policy

import torch
import numpy as np
import Neurosmash

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 64         # Please check the Updates section above for more details
timescale  = 10     # Please check the Updates section above for more details
env = Neurosmash.Environment(timescale=timescale)

# Load VAE weights
vae = VAE(device, image_channels=3).to(device)
vae.load_state_dict(torch.load("vae_v2.torch"))

# Load RNN weights
rnn = MDNRNN(32, 10, 5, 1).to(device)
rnn.load_state_dict(torch.load("rnn.torch"))

# Load controller
controller = Controller().to(device)

# Optimizer
optimizer = torch.optim.Adam(controller.parameters())

def main(episodes):
    # Episode lasts until end == 1
    for episode in range(episodes):
        print("Episode: ", episode)
        end, reward, state = env.reset()  # Reset environment and record the starting state
        done = False
        # Go through every episode but only 15 timesteps
        for time in range(100):

            visual = torch.FloatTensor(state).reshape(size, size, 3) / 255.0
            visual = visual.permute(2, 0, 1)
            encoded_visual = vae.encode(visual.reshape(1, 3, 64, 64).cuda())[0]

            # 3 actions
            futures = []
            for i in range(2):
                action = torch.Tensor(i)
                hidden = rnn.init_hidden(1)
                (pi, mu, sigma), hidden_future = rnn(encoded_visual, hidden)
                futures.append(hidden_future)

            futures = torch.stack(futures)
            print(futures.shape)

            action = select_action(state, controller).detach()
            # Env step
            done, reward, state = env.step(action)
            # Save reward
            if reward > 0:
                print(reward)

            reward += time # reward for existing
            controller.reward_episode.append(reward)

            if done:
                break
        print(reward)

        # Actually backpropagate the policy gradient
        update_policy()

main(5)