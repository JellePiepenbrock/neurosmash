import Neurosmash
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import Adam
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from random import randint

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from torchvision.utils import save_image


from VAE import VAE

ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 64         # Please check the Updates section above for more details
timescale  = 30     # Please check the Updates section above for more details
create_batches = False
batch_size = 128

# vae = VAE(image_channels=3)
# # policy.load_state_dict(torch.load("weights_100episodes_reward_quick_victory"))
# optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
# print('Loaded model.')
#
# # https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb <-- source for code.
# def loss_fn(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return BCE + KLD, BCE, KLD
#
# def select_action():
#     return 3

def select_action():
    return np.random.randint(low=0, high=3)
# def compare(x):
#     recon_x, _, _ = vae(x)
#     return torch.cat([x / 255, recon_x])

def main(episodes):
    env = Neurosmash.Environment(timescale=timescale, size=size, port=port, ip=ip) # This is the main environment.
    end, reward, state = env.reset()
    ep_cnt = 0 
    states = []
    actions = []
    dones = []

    while len(states) < 20:
        epstates = []
        epactions = []
        epdones = []

        end, reward, state = env.reset()  # Reset environment and record the starting state
        done = False
        ep_cnt += 1
        # print(episode)

        for time in range(50):
            state = np.array(state)
            action = select_action()
            epactions.append(action)
            # Step through environment using chosen action
            done, reward, state = env.step(action)
            state = torch.FloatTensor(state).reshape(size, size, 3)
            state = state.permute(2, 0, 1)
            # print(state.shape)
            epstates.append(state)
            epdones.append(done)
            if done:
                print(time)
                break

        print(torch.stack(epstates).shape)
        print(epactions[:20])
        if len(epstates) >= 20:
            states.append(torch.stack(epstates)[-20:])
            actions.append(torch.Tensor(epactions[-20:]))
            dones.append(torch.Tensor(epdones[-20:]))


    torch.save(torch.stack(states), 'training_data.pt')
    torch.save(torch.stack(actions), 'training_actions.pt')
    torch.save(torch.stack(dones), 'training_dones.pt')


main(10)