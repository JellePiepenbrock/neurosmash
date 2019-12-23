from VAE import VAE
from rnn_vae import MDNRNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load VAE weights
vae = VAE(device, image_channels=3).to(device)
vae.load_state_dict(torch.load("vae_v2.torch"))

# Load RNN weights
rnn = MDNRNN().to(device)
rnn.load_state_dict(torch.load("rnn.torch"))

# Load controller
# 32 lv
# 100 hidden_state


# controller =