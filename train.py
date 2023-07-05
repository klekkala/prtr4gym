import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from collections import defaultdict
import imageio as iio
from torch.hub import load_state_dict_from_url
from arguments import get_args
import os
import random
import numpy as np
from pytorch_metric_learning import losses
import math
from IPython.display import clear_output
from PIL import Image
from tqdm import trange, tqdm
import utils
import initial

args = get_args()

encodernet, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)


# %% Start Training
for epoch in trange(start_epoch, args.nepoch, leave=False):
    
    # Start the lr scheduler
    utils.lr_Linear(optimizer, args.nepoch, epoch, args.lr)
    encodernet.train()
    loss_iter = []
    
    #contrastive case: for i, (img_batch1, img_batch2, pair) in enumerate(tqdm(trainloader, leave=False)):
    #img_batch1, img_batch2 -> [B, T, H, W]
    for i, (img, target) in enumerate(tqdm(trainloader, leave=False)):

        image_reshape_val = img.to(device)/div_val
        
        #4STACK: [Bx4xHxW, class]
        #1CHANLSTM: B, class



        if 'CONT' in args.model and 'LSTM' in args.model:

            #4STACK: [Bx4xHxW, B]
            #4STACK: [Bx4xD, B]
            embeddings = encodernet(image_reshape_val)
            loss_func = losses.ContrastiveLoss()
            loss = loss_func(embeddings, target)
            #in the case of lstm: we do temporal unfolding and then compute the embeddings
            #Look into this tomorrow (4th July)
            #1CHANLSTM: [BxTxHxW, BxT]
            #1CHANLSTM: [BxTxD, BxT]
            #1CHANLSTM: [{B+T}xD, {B+T}]

        elif 'CONT' in args.model:
            embeddings = encodernet(image_reshape_val)
            loss_func = losses.ContrastiveLoss()
            loss = loss_func(embeddings, target)

        else:

            targ = target.to(device)/div_val         
            

            recon_data, mu, logvar = encodernet(image_reshape_val)


            #in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
    
            loss = utils.vae_loss(recon_data, targ, mu, logvar, args.kl_weight)            
            #input to the loss function is going to be 2 embeddings and perhaps a bool to see if they belong to the same pairwise distance
            #loss = utils.cont_loss(embeddings, targ, mu, logvar, args.kl_weight)
            
        
        #loss = loss_fn()
        loss_iter.append(loss.item())
        loss_log.append(loss.item())


        #from IPython import embed; embed()
        encodernet.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(np.mean(np.array(loss_iter)))

    # Save a checkpoint with a specific filename
    torch.save({
        'epoch': epoch,
        'loss_log': loss_log,
        'model_state_dict': encodernet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()

    }, args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (args.arch).upper() + "_" + str(args.kl_weight) + "_" + str(args.batch_size) + ".pt")

