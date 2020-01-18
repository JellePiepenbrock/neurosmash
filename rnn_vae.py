import torch.nn as nn
import torch
import torch.nn.functional as F
import Neurosmash
import matplotlib
from VAE import VAE
from sklearn.preprocessing import scale
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
# from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
# from vae import VAE
from sklearn.decomposition import PCA
from torchvision.utils import save_image
# from IPython.core.display import Image, display
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 64         # Please check the Updates section above for more details
timescale  = 10     # Please check the Updates section above for more details
# env = Neurosmash.Environment(timescale=timescale)

weighted_loss = 1

# Load VAE weights

vae = VAE(device, image_channels=3).to(device)
vae.load_state_dict(torch.load("data_folder_vae/vae_v3_weighted_loss_{}.torch".format(weighted_loss)))
vae = vae.to(device)
ds_size = 2000
bsz = 100
epochs = 100
seqlen = 10
n_layers = 1
n_gaussians = 5

z_size = 32
n_hidden = 256
n_gaussians = 5


# z2 = torch.load('data_folder_vae/training_data_encoded_weighted_loss_{}.pt'.format(weighted_loss)).to(device)
# aa = torch.load('data_folder_vae/training_actions.pt').to(device)
# print(z2.shape)
# print(torch.Tensor(aa))
# print(aa.shape)
# file = open('PCA_model.pkl','r')
# pca = pickle.load(file)
# for episode in z2:
#     while len(episode) < 15:
#         episode.append(torch.zeros(3, 64, 64))
#
#     while len(episode) > 15:
# #         episode = episode[:15]
#
# print(z2[0].shape)
# print(torch.tensor(z2[0]))

# pca = PCA(n_components=z_size)
# pca = torch.load("pca.pt")

# print(z2.shape)
# print(pca)
# z2_pca = z2.reshape(20*20, 3*64*64) / 1000
# print(z2.shape)
# z = torch.Tensor(pca.fit_transform(z2_pca).reshape(1000, 20, z_size)).cuda()
# z = torch.Tensor(pca.transform(z2_pca).reshape(20, 20, z_size)).cuda()
# # torch.save(pca, "pca.pt")
# print(z.shape)
# # THIS LINE DOES LIKE 12 THINGS
#
# z2 = z2.reshape(ds_size, 20, 32)
# print(z2.shape, aa.shape)
# z = torch.cat([z2[:, 1:, :], aa.reshape(ds_size, 20, 1)[:, :-1, :]], dim=2)
# print(z)
# print(z.shape)
# print(pca.shape)

# z = torch.randn(200, 150, 4).cuda()
# z = z.to(device)

def reduce_logsumexp(x, pi, dim=None):

    """"
    This function is more numerically stable than naively taking the log sum exp for the MDN loss. The dim should be the dimension
    of the Gaussians. If you have 5 gaussians, dim should be the dimension with size 5.

    Why does this work? https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

    I added pi in this function so that the model itself could stay unchanged in test instances
    """

    max_x, _ = x.max(dim=dim, keepdim=True) # get the max value
    y = ((x - max_x).exp()*pi).sum(dim=dim).log() # do the trick to avoid numerical instability

    return max_x + y.unsqueeze(2) # make sure y is of correct shape!

def detach(states):
    return [state.detach() for state in states]


# Mixed Density RNN following https://github.com/sksq96/pytorch-mdn
# Added action inputs  etc
class MDNRNN(nn.Module):
    def __init__(self, z_size, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers

        self.lstm = nn.LSTM(z_size + 1, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians * (z_size))
        self.fc2 = nn.Linear(n_hidden, n_gaussians * (z_size))
        self.fc3 = nn.Linear(n_hidden, n_gaussians * (z_size))

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)

        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)

        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma

    def forward(self, x, h):
        # Forward propagate LSTM
        # print(x.shape)
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)

    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))


model = MDNRNN(z_size, n_hidden, n_gaussians, n_layers).to(device)


def mdn_loss_fn(y, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # loss = torch.exp(m.log_prob(y))
    # # print("explogprob", loss)
    # loss = torch.sum(loss * pi, dim=2)
    # loss = -torch.log(loss)

    # Use the stable version of this loss
    loss = - reduce_logsumexp(m.log_prob(y), pi,
                              dim=2)  # dim 2 is the k=5 dimension, so we get the loss at each time step at each batch

    return loss.mean()

def criterion(y, pi, mu, sigma):
    y = y.unsqueeze(2)
    return mdn_loss_fn(y, pi, mu, sigma)

if __name__ == "main":

    optimizer = torch.optim.Adam(model.parameters())

    best_loss = np.inf
    early_stopping = 0
    for epoch in range(epochs):
        if early_stopping > 5:
            print("No improvement for 5 epochs, stopping early")
            break
        epochloss = 0

        z = z[torch.randperm(2000)]
        b = 0
        batch = z[b:b + bsz, :, :]
        while b < 2000:

            hidden = model.init_hidden(bsz)

            for i in range(0, batch.size(1) - seqlen, seqlen):

                inputs = batch[:, i:i + seqlen, :]

                targets = batch[:, (i + 1):(i + 1) + seqlen, :-1]
                hidden = detach(hidden)
                (pi, mu, sigma), hidden = model(inputs, hidden)
                loss = criterion(targets, pi, mu, sigma)


                model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                epochloss += loss.item()

            b += bsz
        if epoch % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch, epochs, epochloss))

        if epochloss < best_loss:
            best_loss = epochloss
            early_stopping = 0
        elif best_loss > epochloss:
            early_stopping += 1


    torch.save(model.state_dict(), "rnn_29dec_{}.torch".format(weighted_loss))

    # print(z.shape)
    x = z[0, 10, :].reshape(1, 1, z_size+1)
    # print(x.shape)
    hidden = model.init_hidden(1)
    (pi, mu, sigma), _ = model(x, hidden)
    print(pi)
    original_x = vae.decode(x[:, :, :z_size].reshape(1, 1, 32)).reshape(3, 64, 64)
    original_x = (original_x * 255)
    # print(original_x.shape)
    matplotlib.pyplot.imshow(np.transpose(np.array(original_x.detach().cpu().numpy(), "uint8"), (1, 2, 0)))
    plt.title("Original")
    plt.show()

    x = z[0, 11, :].reshape(1, 1, z_size+1)
    # print(x.shape)
    hidden = model.init_hidden(1)
    # (pi, mu, sigma), _ = model(x, hidden)
    print(pi)
    original_x = vae.decode(x[:, :, :z_size].reshape(1, 1, 32)).reshape(3, 64, 64)
    original_x = (original_x * 255)
    # print(original_x.shape)
    matplotlib.pyplot.imshow(np.transpose(np.array(original_x.detach().cpu().numpy(), "uint8"), (1, 2, 0)))
    plt.title("Original")
    plt.show()
    # print(y_preds.shape)

    for i in range(5):
        y_preds = torch.normal(mu, sigma)[:, :, i, :]
        print(y_preds.shape)
        compare_x = vae.decode(y_preds).reshape(3, 64, 64)
        compare_x = (compare_x * 255)


        matplotlib.pyplot.imshow(np.transpose(np.array(compare_x.detach().cpu().numpy(), "uint8"), (1, 2, 0)))
        plt.title("Possible Future {}".format(i))
        plt.show()
    # # compare_x = vae.decode(z_out)
