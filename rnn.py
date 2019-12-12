import torch.nn as nn
import torch
import torch.nn.functional as F
import Neurosmash
import matplotlib
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
ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 64         # Please check the Updates section above for more details
timescale  = 10     # Please check the Updates section above for more details
env = Neurosmash.Environment(timescale=timescale)

bsz = 10
epochs = 500
seqlen = 1
n_layers = 2
n_gaussians = 5

z_size = 200
n_hidden = 100
n_gaussians = 5


z2 = torch.load('training_data.pt')
aa = torch.load('training_actions.pt')

print(torch.Tensor(aa))
print(aa.shape)
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
pca = torch.load("pca.pt")

# print(z2.shape)
# print(pca)
z2_pca = z2.reshape(20*20, 3*64*64) / 1000
# print(z2.shape)
# z = torch.Tensor(pca.fit_transform(z2_pca).reshape(1000, 20, z_size)).cuda()
z = torch.Tensor(pca.transform(z2_pca).reshape(20, 20, z_size)).cuda()
# torch.save(pca, "pca.pt")
print(z.shape)
# THIS LINE DOES LIKE 12 THINGS

z = torch.cat([z[:, 1:, :], aa.reshape(20, 20, 1).cuda()[:, :-1, :]], dim=2)
print(z)
print(z.shape)
# print(pca.shape)

# z = torch.randn(200, 150, 4).cuda()



def detach(states):
    return [state.detach() for state in states]

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
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)

    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))


model = MDNRNN(z_size, n_hidden, n_gaussians, n_layers).to(device)

def mdn_loss_fn(y, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    # print("explogprob", loss)
    loss = torch.sum(loss * pi, dim=2)
    loss = -torch.log(loss)
    return loss.mean()

def criterion(y, pi, mu, sigma):
    y = y.unsqueeze(2)
    return mdn_loss_fn(y, pi, mu, sigma)

optimizer = torch.optim.Adam(model.parameters())


for epoch in range(epochs):
    z = z[torch.randperm(20)]
    b = 0
    # Set initial hidden and cell states
    hidden = model.init_hidden(bsz)
    # print(hidden[0].shape)
    # print(z.shape)
    # print(z.size(1) - seqlen)
    # print(seqlen)
    batch = z[b:b+bsz, :, :]
    for i in range(0, batch.size(1) - seqlen, seqlen):
        # Get mini-batch inputs and targets
        inputs = batch[:, i:i + seqlen, :]

        targets = batch[:, (i + 1):(i + 1) + seqlen, :-1]
        # print(targets.shape)
        # Forward pass
        hidden = detach(hidden)
        (pi, mu, sigma), hidden = model(inputs, hidden)
        # print(targets)
        loss = criterion(targets, pi, mu, sigma)

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch, epochs, loss.item()))
    b += bsz
# zero = np.random.randint(z.size(0))
# one = np.random.randint(z.size(1))
# x = z[zero:zero+1, one:one+1, :]
# y = z[zero:zero+1, one+1:one+2, :]
for k in range(19):
    x = z[0, k, :].reshape(1, 1, z_size + 1)
    original_x = pca.inverse_transform(x[:, :, :z_size].cpu().detach().numpy()).reshape(3, 64, 64) * 1000
    matplotlib.pyplot.imshow(np.transpose(np.array(original_x, "uint8"), (1, 2, 0)))
    plt.show()

x = z[0, 10, :].reshape(1, 1, z_size+1)
print(x.shape)
hidden = model.init_hidden(1)
(pi, mu, sigma), _ = model(x, hidden)
original_x = pca.inverse_transform(x[:, :, :z_size].cpu().detach().numpy()).reshape(3, 64, 64) * 1000
matplotlib.pyplot.imshow(np.transpose(np.array(original_x, "uint8"), (1, 2, 0)))
plt.title("Original")
plt.show()
# print(y_preds.shape)

for i in range(5):
    y_preds = torch.normal(mu, sigma)[:, :, i, :]
    compare_x = pca.inverse_transform(y_preds.cpu().detach().numpy()).reshape(3, 64, 64) * 1000


    matplotlib.pyplot.imshow(np.transpose(np.array(compare_x, "uint8"), (1, 2, 0)))
    plt.title("Possible Future {}".format(i))
    plt.show()
# compare_x = vae.decode(z_out)
