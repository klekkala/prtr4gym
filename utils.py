from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from IPython import embed
from pytorch_metric_learning import losses
from arguments import get_args
import numpy as np
import math
from torch.distributions.normal import Normal

args = get_args()


class PadSequence:
    def __call__(self, batch):
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.Tensor([len(x[0]) for x in sorted_batch])
        max_seq_length = int(lengths[0].numpy())
        # Get each sequence and pad it
        obs_sequences = [x[0] for x in sorted_batch]
        obs_sequences_padded = np.array(
            [np.concatenate((x, np.tile(x[-1], (max_seq_length - len(x), 1, 1, 1)))) for x in obs_sequences])  # padding
        action_sequences = [torch.tensor(x[1]) for x in sorted_batch]
        action_sequences_padded = torch.nn.utils.rnn.pad_sequence(action_sequences, batch_first=True)
        target_sequences = [x[1:] for x in obs_sequences_padded]
        target_sequences_padded = np.array(
            [np.concatenate((x, np.tile(x[-1], (1, 1, 1, 1)))) for x in target_sequences])
        return (
        torch.tensor(obs_sequences_padded), action_sequences_padded, lengths, torch.tensor(target_sequences_padded))


def get_data_STL10(negset, neg_batch_size, transform, posset=None, pos_batch_size=None):
    if negset != None:
        print("Loading trainset...")
        if 'LSTM' in args.model:
            negloader = DataLoader(negset, batch_size=neg_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        else:
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
    # recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    # the below was what was used for the prev experiment that was working
    recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction='mean')
    # recon_loss = F.mse_loss(torch.sigmoid(recon), x, reduction='mean')
    KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_weight * (KL_loss)
    return loss


def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob



def clip_loss(fpv_embed, bev_embed, temperature, labels):
    
    #normalize the proj embeddings
    nfpv_embed = nn.functional.normalize(fpv_embed, p=2, dim=1)
    nbev_embed = nn.functional.normalize(bev_embed, p=2, dim=1)

    logits = torch.matmul(nfpv_embed, nbev_embed.T) * math.exp(temperature)
    #logits = torch.matmul(nfpv_embed, nbev_embed.T).cuda()

    #chen change cross_entropy_loss to the pytorch one
    loss_fpv = F.cross_entropy(logits.T, labels.cuda())
    loss_bev = F.cross_entropy(logits, labels.cuda())
    
    return (loss_fpv + loss_bev)/2


    
    
def vip_loss(start_embed, mid_embed, midplus_embed, end_embed):
    
    V_0 = -torch.linalg.norm(start_embed-end_embed)
    V_t1 = -torch.linalg.norm(mid_embed-end_embed)
    V_t2 = -torch.linalg.norm(midplus_embed-end_embed)
    VIP_loss = (1-args.sgamma)*-V_0.mean() + torch.logsumexp(V_t1+1-args.sgamma*V_t2, dim=-1)
    return VIP_loss