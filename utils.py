
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

def get_data_STL10(trainset, testset, transform, batch_size):
    
    if trainset != None:
        print("Loading trainset...")
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if testset != None:
        print("Loading testset...")
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    print("Done!")
    if trainset == None:
        return None, testloader
    if testset == None:
        return trainloader, None


# Linear scaling the learning rate down
def lr_Linear(optimizer, epoch_max, epoch, lr):
    lr_adj = ((epoch_max - epoch) / epoch_max) * lr
    set_lr(optimizer, lr=lr_adj)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vae_loss(recon, x, mu, logvar, kl_weight):
    #recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    #the below was what was used for the prev experiment that was working
    recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction='mean')
    #recon_loss = F.mse_loss(torch.sigmoid(recon), x, reduction='mean')
    KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_weight*KL_loss
    return loss
