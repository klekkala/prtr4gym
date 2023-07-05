from RES_VAE import VAE as VAE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils

use_cuda = torch.cuda.is_available()
print(use_cuda)
GPU_indx = 0
device = torch.device(GPU_indx if use_cuda else "cpu")
print(device)

vae_net = VAE(channel_in=3, ch=64).to(device)
checkpoint = torch.load("Models/" + "STL10_ATTARI_64.pt" , map_location="cpu")
print("Checkpoint loaded")
vae_net.load_state_dict(checkpoint['model_state_dict'])

_ , mu, sigma = vae_net(obs.to(device)))
torch.distributions.Normal(loc=mu, scale=sigma)
