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
import cv2
import numpy as np
from matplotlib import pyplot as plt
from VAE import VAE


from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor
device = torch.device('cuda')

batch_size = 32

vae = VAE(image_channels=3).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
print('Loaded model.')

mode = 'filter'

# https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb <-- source for code. 
def loss_fn(recon_x, x, vae_weights, mu, logvar):
    # Generate weights

    # Weighted BCE
    # vae_weights = torch.ones(1, 3, 64, 64).to(device)
    # vae_weights[:,0,:,:] *= 0.001
    # vae_weights *= 0.99
    # vae_weights = vae_weights.repeat(recon_x.shape[0], 1, 1, 1)
    
    BCE = F.binary_cross_entropy(recon_x, x, weight=vae_weights, reduction='sum')
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def compare(x):
    recon_x, _, _ = vae(x)
    return torch.cat([x, recon_x])

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
    '''
    TODO:
    Assign weights based on distribution pixels.

    ADD Filters as weights, but set the zeros to 0.001 and ones to 0.99.

    Can also just use normal images, but with the CANNY edge detection as weights.
    
    '''
    dataset = None
    for i in range(5):
        if dataset is None:
            dataset = torch.load('{}/data/training_data_{}.pt'.format(base_url, i)) / 255
        else:
            dataset = torch.cat((dataset, torch.load('{}/data/training_data_{}.pt'.format(base_url, i)) / 255))
    
    test_dataset = torch.load('{}/data/training_data_{}.pt'.format(base_url, i+1)) / 255
    dataset = np.array([np.transpose(x.data.numpy(), (1,2,0)) for x in dataset])  

    if mode == 'ext':
        dataset = [max_rgb_filter(np.uint8(x*255)) for x in dataset]
        dataset = np.transpose(dataset, (0, 3, 1, 2))
        dataset[:,0,:,:] *= 0
    elif mode == 'canny':
        filters = [cv2.Canny(np.uint8(x_*255*255), 100, 200) for x_ in dataset]
        dataset = np.array([cv2.bitwise_and(x_, x_, mask=f) for x_, f in zip(dataset, filters)])
        dataset = np.transpose(dataset, (0, 3, 1, 2))
    elif mode == 'filter':
        filters = np.repeat(np.array([np.array(cv2.Canny(np.uint8(x_*255*255), 100, 200)) for x_ in dataset]).reshape(-1, 1, 64, 64), 3, axis=1)
        dataset = np.transpose(dataset, (0, 3, 1, 2))
        dataset = np.array([np.array([x, f]) for x,f in zip(dataset, filters)])

    dataset = torch.from_numpy(dataset).float() #/ 255

    # dataset = dataset[:100]
    # # Divide by 255
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = 3000
    print(dataset.shape, test_dataset.shape)
    for epoch in range(epochs):
        losses = [] 
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            f = (batch[:,1,:,:,:].squeeze()).to(device)
            f[f == 0] = 0.05
            f[f == 255] = 0.95

            # torch.set_printoptions(profile="full")
            # print(f[0])
            # print('---')
            # torch.set_printoptions(profile="default")
            # break

            b = batch[:,0,:,:,:].squeeze().to(device)
            recon_images, mu, logvar = vae(b)
            
            loss, bce, kld = loss_fn(recon_images, b, f, mu, logvar)
            loss.backward()
            optimizer.step()

        if epoch == 100:
            print('reducing lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        if epoch == 200:
            print('reducing lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        print('Epoch {}, loss {}'.format(epoch+1, loss.item() / batch_size))
        if epoch % 5 == 0:
            # Store model.
            torch.save(vae.state_dict(), '{}/neurosmash/vae_v2.torch'.format(base_url))
            
            fixed_x = b[randint(0, len(b)-1)].unsqueeze(0)
            compare_x = compare(fixed_x.to(device))
            
            # print(torch.stack(, compare_x).shape)
            save_image(compare_x.data.cpu(), '{}/neurosmash/images/sample_image_epoch_{}.png'.format(base_url, epoch))

            # Show final result. 
            img = mpimg.imread('{}/neurosmash/images/sample_image_epoch_{}.png'.format(base_url, epoch))
            imgplot = plt.imshow(img.squeeze())
            plt.show()
    
    print('Predicting test set:')
    for i, batch in enumerate(test_dataloader):
        if i > 10:
            break
        b = b.to(device)
        recon_images, mu, logvar = vae(b)

        fixed_x = b[randint(0, len(b)-1)].unsqueeze(0)
        compare_x = compare(fixed_x.to(device))
        
        save_image(compare_x.data.cpu(), '{}/neurosmash/images/sample_image_test_epoch_{}.png'.format(base_url, epoch))

        # Show final result.
        img = mpimg.imread('{}/neurosmash/images/sample_image_test_epoch_{}.png'.format(base_url, epoch))
        imgplot = plt.imshow(img.squeeze())
        plt.show()

    # Store model.
    torch.save(vae.state_dict(), '{}/neurosmash/vae_v2.torch'.format(base_url))

main(50)