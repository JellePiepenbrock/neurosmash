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
size       = 64         # Please check the Updates section above for more details
timescale  = 30     # Please check the Updates section above for more details
create_batches = False
batch_size = 32

env = Neurosmash.Environment(timescale=timescale, size=size, port=port, ip=ip) # This is the main environment.
end, reward, state = env.reset()
vae = VAE(image_channels=3)
# policy.load_state_dict(torch.load("weights_100episodes_reward_quick_victory"))
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
print('Loaded model.')

# https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb <-- source for code. 
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean') #TODO What does size_average do..?
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def select_action():
    return 3

def main(episodes):
    #TODO: Sample average?
    #TODO: Make the VAE less complex; reduce image size.
    '''
    The code below creates its own batches and automatically trains on these
    batches in a live modus.
    '''
    if create_batches:
        ep_cnt = 0 
        batches = []
        for i in range(1000): 
            states = []
            for episode in range(episodes):
                end, reward, state = env.reset()  # Reset environment and record the starting state
                done = False
                ep_cnt += 1

                for time in range(1000):
                    state = np.array(state)
                    action = select_action()
                    # Step through environment using chosen action
                    done, reward, state = env.step(action)
                    state = torch.FloatTensor(state).reshape(size, size, 3)
                    state = state.permute(2, 0, 1)
                    states.append(state)

                    if done:
                        break
                
                if len(states) > batch_size:
                    break
            states = torch.stack(states)[:batch_size]
            batches.append(states)
            states = []
            print('Added another batch: {}'.format(len(batches)))
            print('Wut, len states: {}'.format(len(states)))
            print('-----------------------------')
            if len(batches) == 1000:
                batches = torch.stack(batches)
                torch.save(batches, 'file.pt')
                break
    batches = torch.load('file.pt')
    for i in range(50):
        t_loss = 0
        for i, b in enumerate(batches):
            b = b / 255
            recon_images, mu, logvar = vae(b)
            loss, bce, kld = loss_fn(recon_images, b, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()

            if i+1 % 50 == 0:
                print('Current avg batch loss: {}'.format(t_loss / i))
            # Sample image
            #TODO: Images should be rounded.

            # sample = recon_images[0].data.cpu() / 255
            # sample = sample.astype(np.uint8)
            # sample = torch.from_numpy(sample)
            # sample_orig = b[0].data.cpu() / 255

            # save_image(sample, './vae_images/sample_image.png')
            # save_image(sample_orig, './vae_images/orig_image.png')


            # display(Image('sample_image.png', width=700, unconfined=True))
                # to_print = "Episode count[{}] Loss: {:.3f} {:.3f} {:.3f}".format(ep_cnt, loss.data[0]/bs, bce.data[0]/bs, kld.data[0]/bs)
                # print(to_print)
        print('Loss for this epoch: {}'.format(t_loss / len(batches)))
        print('----------------------------------------------------')


main(50)