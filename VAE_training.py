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

from torchvision.utils import save_image


from VAE import VAE

ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 768         # Please check the Updates section above for more details
timescale  = 30     # Please check the Updates section above for more details

# agent = Neurosmash.Agent() # This is an example agent.
                           # It has a step function, which gets reward/state as arguments and returns an action.
                           # Right now, it always outputs a random action (3) regardless of reward/state.
                           # The real agent should output one of the following three actions:
                           # none (0), left (1) and right (2)

# agent = Neurosmash.RLAgent()
env = Neurosmash.Environment(timescale=timescale) # This is the main environment.
end, reward, state = env.reset()
vae = VAE(image_channels=3)
# policy.load_state_dict(torch.load("weights_100episodes_reward_quick_victory"))
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)


# https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb <-- source for code. 
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def select_action():
    return 3

def main(episodes):
    ep_cnt = 0 
    for episode in range(episodes):
        print("Episode: ", episode)
        end, reward, state = env.reset()  # Reset environment and record the starting state
        done = False
        ep_cnt += 1
        states = []
        for time in range(15):
            state = np.array(state)
            action = select_action()
            # Step through environment using chosen action
            done, reward, state = env.step(action)
            state = torch.FloatTensor(state).reshape(768, 768, 3)
            state = state.permute(2, 0, 1)
            states.append(state)

            if done:
                break
        states = torch.stack(states)
        recon_images, mu, logvar = vae(states)
        loss, bce, kld = loss_fn(recon_images, states, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

        # Sample image
        sample = recon_images[0]

        #TODO: Does not work yet, need to change the images.
        save_image(sample.data.cpu(), './vae_images/sample_image.png')
        save_image(state.data.cpu(), './vae_images/orig_image.png')

        display(Image('sample_image.png', width=700, unconfined=True))
            # to_print = "Episode count[{}] Loss: {:.3f} {:.3f} {:.3f}".format(ep_cnt, loss.data[0]/bs, bce.data[0]/bs, kld.data[0]/bs)
            # print(to_print)


main(50)