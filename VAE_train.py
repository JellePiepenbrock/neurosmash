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
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from matplotlib import pyplot as plt
from VAE import VAE

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor

'''
Code based on https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
credits for providing example VAE model go to respective author.
'''

base_url = './'
type_device = 'cpu'
device = torch.device(type_device)
batch_size = 32
vae = VAE(device, image_channels=3).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def loss_fn(recon_x, x, vae_weights, mu, logvar, use_weights):
    # Generate weights

    if use_weights:
        BCE = F.binary_cross_entropy(recon_x, x, weight=vae_weights, reduction='sum').div(batch_size)
    else:
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum').div(batch_size)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def compare(x, f, vae):
    vae.eval()
    recon_x, _, _ = vae(x)
    vae.train()
    return torch.cat([x, recon_x, f])

def max_rgb_filter(image):
    (R, G, B) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    return cv2.merge([R, G, B])
 
def main(episodes):
    dataset = torch.load('{}/training_data.pt'.format(base_url)) / 255
    dataset_flattened = np.array([np.transpose(x.data.numpy(), (1,2,0)) for x in dataset.reshape(-1, 3, 64, 64)])

    backSub = cv2.createBackgroundSubtractorMOG2()
    temp = np.uint8(dataset_flattened*255)[:10000]
        
    for frame in temp:   
        frame = frame[0,:,:]
        fgMask = backSub.apply(frame)

    filters = np.repeat(np.array([np.array(backSub.apply(np.uint8(x_*255))) for x_ in dataset_flattened]).reshape(-1, 1, 64, 64), 3, axis=1)
    dataset_flattened = np.transpose(dataset_flattened, (0, 3, 1, 2))
    dataset_flattened = np.array([np.array([x, f]) for x,f in zip(dataset_flattened, filters)])
    dataset_flattened = torch.from_numpy(dataset_flattened).float()
    dataloader = torch.utils.data.DataLoader(dataset_flattened, batch_size=batch_size, shuffle=True)

    for use_weights in [0, 1]:
        print('===========================')
        print('Training new model with weights: {}'.format(use_weights))
        print('===========================')
        vae = VAE(image_channels=3).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        vae.train()
        print('Loaded model.')
        for epoch in range(epochs):
            losses = [] 
            batch_loss = 0
            
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                f = batch[:,1,:,:,:].squeeze().to(device)
                f[f > 0] = 5
                f[f == 0] = 0.5

                b = batch[:,0,:,:,:].squeeze().to(device)
                recon_images, mu, logvar = vae(b)
                
                loss, bce, kld = loss_fn(recon_images, b, f, mu, logvar, use_weights)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            batch_loss /= len(dataloader)
            if batch_loss < best_loss:
                print('Storing weights', batch_loss, best_loss)
                best_loss = batch_loss
                torch.save(vae.state_dict(), '{}/neurosmash/vae_v3_weights_{}.torch'.format(base_url, use_weights))

            if epoch == 50:
                print('reducing lr')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4
            elif epoch == 100:
                print('reducing lr')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
            
            print('Epoch {}, loss {}'.format(epoch+1, batch_loss / batch_size))

            if epoch % 5 == 0:
                # Store model. 
                idx = randint(0, len(b)-1)           
                fixed_x = b[idx].unsqueeze(0)
                fixed_filter = batch[:,1,:,:,:].squeeze()[idx].unsqueeze(0)
                compare_x = compare(fixed_x.to(device), fixed_filter.to(device), vae)
                
                # print(torch.stack(, compare_x).shape)
                save_image(compare_x.data.cpu(), '{}/img_vae_{}/sample_image_epoch_{}.png'.format(base_url, use_weights, epoch))

                # Show final result. 
                img = mpimg.imread('{}/img_vae_{}/sample_image_epoch_{}.png'.format(base_url, use_weights, epoch))
                imgplot = plt.imshow(img.squeeze())
                plt.show()
main(50)