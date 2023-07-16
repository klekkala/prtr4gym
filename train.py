import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Datasets
from itertools import cycle
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.autograd import Variable
from collections import defaultdict
from pytorch_metric_learning import losses
import imageio as iio
#from fancylosses import InfoNCE
from IPython import embed
from torch.hub import load_state_dict_from_url
from arguments import get_args
#from info_nce import InfoNCE
import os
import random
import numpy as np

import math
from IPython.display import clear_output
from PIL import Image
from tqdm import trange, tqdm
import utils
import initial

args = get_args()

if 'DUAL' in args.model:
    encodernet, teachernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)
elif 'CONT' in args.model:
    encodernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)
else:
    encodernet, negloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)


#select which contrastive loss to train
if 'CONT' in args.model:
    #loss_func = InfoNCE(temperature=args.temperature, negative_mode='unpaired') # negative_mode='unpaired' is the default value
    loss_func = losses.ContrastiveLoss()
elif 'LSTM' in args.model:
    loss_func = nn.MSELoss()
else:
    loss_func = utils.vae_loss

# %% Start Training
for epoch in trange(start_epoch, args.nepoch, leave=False):
    
    # Start the lr scheduler
    utils.lr_Linear(optimizer, args.nepoch, epoch, args.lr)
    encodernet.train()
    loss_iter = []
    
    #contrastive case: for i, (img_batch1, img_batch2, pair) in enumerate(tqdm(trainloader, leave=False)):
    #img_batch1, img_batch2 -> [B, T, H, W]

    if 'CONT' in args.model:
        positerator = iter(posloader)
        
    for i, negdata in enumerate(tqdm(negloader, leave=False)):
    
        if 'CONT' in args.model:
            try:
                posdata = next(positerator)
            except StopIteration:
                positerator = iter(posloader)
                posdata = next(dataloader_iterator)
                #(img, value, episode) = data
                #img = img.reshape(img.shape[0]*img.shape[1], img.shape[2], img.shape[3], img.shape[4])
                #value = torch.flatten(value)
                #episode = torch.flatten(episode)
                #target = torch.cat((torch.unsqueeze(value, 1), torch.unsqueeze(episode, 1)), axis=1)
            neg_reshape_val = negdata.to(device)/div_val
            pos_reshape_val = posdata.to(device)/div_val
            
            #batch_size, num_negative, embedding_size = 32, 48, 128
            #sample_batch_size -> number of positives/queries
            #train_batch_size -> number of negatives
            query_imgs, pos_imgs = torch.split(pos_reshape_val, 1, dim=1)
            query_imgs = torch.squeeze(query_imgs)
            pos_imgs = torch.squeeze(pos_imgs)
            
            query = encodernet(query_imgs)
            positives = encodernet(pos_imgs)
            negatives = encodernet(neg_reshape_val)


            #allocat classes for queries, positives and negatives
            posclasses = torch.arange(start=0, end=query.shape[0])
            negclasses = torch.arange(start=query.shape[0], end=query.shape[0]+negatives.shape[0])
            
            if 'DUAL' in args.model:
                #for now we are testing with the same game
                #so the adapter learns an identity function
                tquery = teachernet(query_imgs)
                tpositives = teachernet(pos_imgs)
                tnegatives = teachernet(neg_reshape_val)
                loss = loss_func(torch.cat([query, positives, negatives, tquery, tpositives], axis=0), torch.cat([posclasses, posclasses, negclasses, posclasses, posclasses], axis=0))
            else:
                loss = loss_func(torch.cat([query, positives, negatives], axis=0), torch.cat([posclasses, posclasses, negclasses], axis=0))

        elif 'LSTM' in args.model:
            #CHEN
            (img, target, action) = negdata
            image_reshape_val = img.to(device)/div_val
            targ = target.to(device)/div_val
            action = action.to(device)

            encodernet.init_hs()
            z_gt, _, _ = encodernet.encode(targ)
            z_prev, _, _ = encodernet.encode(image_reshape_val)
            z_pred = encodernet(action, z_prev)
            loss = loss_func(z_pred, z_gt)

        else:
            (img, target) = negdata
            image_reshape_val = img.to(device)/div_val
            targ = target.to(device)/div_val


            recon_data, mu, logvar = encodernet(image_reshape_val)

            #in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
    
            loss = loss_func(recon_data, targ, mu, logvar, args.kl_weight)
            
        

        loss_iter.append(loss.item())
        loss_log.append(loss.item())
        #from IPython import embed; embed()
        encodernet.zero_grad()
        loss.backward()
        optimizer.step()
    







    print(np.mean(np.array(loss_iter)))
    if 'VAE' in args.model:
        auxval = args.kl_weight
    else:
        auxval = args.temperature
    # Save a checkpoint with a specific filename
    torch.save({
        'epoch': epoch,
        'loss_log': loss_log,
        'model_state_dict': encodernet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()

    }, args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (args.arch).upper() + "_" + str(auxval) + "_" + str(args.sgamma) + "_" + str(args.train_batch_size) + "_" + str(args.sample_batch_size) + ".pt")


