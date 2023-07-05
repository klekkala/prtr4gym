
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



def cont_loss(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss
    """
    #if d == 0:
    #  return T.mean(T.pow(euc_dist, 2))  # distance squared
    #else:  # d == 1
    #  delta = self.m - euc_dist  # sort of reverse distance
    #  delta = T.clamp(delta, min=0.0, max=None)
    #  return T.mean(T.pow(delta, 2))  # mean over all rows


    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    # >>> input1 = torch.randn(100, 128)
    # >>> input2 = torch.randn(100, 128)
    # >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # >>> output = cos(input1, input2)
    return loss


"""
class ContrastiveLoss(T.nn.Module):
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    
    euc_dist = T.nn.functional.pairwise_distance(y1, y2)

    if d == 0:
      return T.mean(T.pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = self.m - euc_dist  # sort of reverse distance
      delta = T.clamp(delta, min=0.0, max=None)
      return T.mean(T.pow(delta, 2))  # mean over all rows
"""
