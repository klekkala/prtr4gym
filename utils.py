
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from IPython import embed
from pytorch_metric_learning import losses
from arguments import get_args

args = get_args()

def get_data_STL10(negset, neg_batch_size, transform, posset=None, pos_batch_size=None):
    
    if negset != None:
        print("Loading trainset...")
        negloader = DataLoader(negset, batch_size=neg_batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if posset != None:
        print("Loading testset...")
        posloader = DataLoader(posset, batch_size=pos_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    print("Done!")
    if negset != None and posset != None:
        return negloader, posloader
    if negset == None:
        return None, posloader
    if posset == None:
        return negloader, None


# Linear scaling the learning rate down
def lr_Linear(optimizer, epoch_max, epoch, lr):
    lr_adj = ((epoch_max - epoch) / epoch_max) * lr
#    lr_adj = 0.1 ** (epoch/10) * lr
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
