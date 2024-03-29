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
from evaluate import eval_adapter
from collections import defaultdict
from pytorch_metric_learning import losses
import imageio as iio
# from fancylosses import InfoNCE
from IPython import embed
from torch.hub import load_state_dict_from_url
from arguments import get_args
# from info_nce import InfoNCE
import os
import random
import numpy as np

import math
from IPython.display import clear_output
from PIL import Image
from tqdm import trange, tqdm
import utils
import initial
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import cv2
args = get_args()


def log_write(encodernet, mode):
    f = open(args.texpname + '_' + args.expname + '.txt', mode)
    if mode == 'w':
        f.write('Before Training\n')
    elif mode == 'a':
        f.write('After Training\n')
    f.close()

    res = []
    for _ in range(args.nrounds):
        reward_val = eval_adapter(
            '/lab/kiran/logs/rllib/atari/4stack/1.a_AirRaidNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_01_30_05/checkpoint/',
            '/lab/kiran/logs/rllib/atari/4stack/1.a_NameThisGameNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_09_14_07/checkpoint/',
            "NameThisGameNoFrameskip-v4", encodernet)
        res.append(reward_val)
    average = sum(res) / len(res)
    print(average)
    f = open(args.texpname + '_' + args.expname + '.txt', 'a')
    f.write(' Average: ' + str(average) + ', ')
    f.write('\nAfter Training\n')
    f.close()


if 'DUAL' in args.model:
    # negloader -> tset, #posloader -> eset
    # len(tset) > len(eset)
    # negloader should be loading the teacher dataset
    encodernet, teachernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(
        True)
elif 'CONT' in args.model:
    encodernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(
        True)
else:
    encodernet, negloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)

# select which contrastive loss to train
if 'CONT' in args.model:
    # loss_func = InfoNCE(temperature=args.temperature, negative_mode='unpaired') # negative_mode='unpaired' is the default value
    loss_func = losses.ContrastiveLoss()
elif 'LSTM' in args.model and 'CARLA' in args.model:
    #loss_func = nn.MSELoss()
    loss_func = utils.gmm_loss
else:
    loss_func = utils.vae_loss

if 'DUAL' in args.model:
    log_write(encodernet, 'w')

if 'LSTM' in args.model and 'CARLA' in args.model:
    print("lksjlk;fdlkasjlkdfj;lasj;lkdfjlkj")
    latent_cls = []   # contains representations of 10 bev classes
    for i in range(10):
        img = cv2.imread("/lab/kiran/img2cmd/manual_label/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=(0, 1))

        image_val = torch.tensor(img).to(device) / div_val
        z, _, _ = encodernet.encode(image_val)
        latent_cls.append(z)

# %% Start Training
for epoch in trange(start_epoch, args.nepoch, leave=False):

    # Start the lr scheduler
    utils.lr_Linear(optimizer, args.nepoch, epoch, args.lr)
    encodernet.train()
    loss_iter = []

    # contrastive case: for i, (img_batch1, img_batch2, pair) in enumerate(tqdm(trainloader, leave=False)):
    # img_batch1, img_batch2 -> [B, T, H, W]

    if 'CONT' in args.model:
        positerator = iter(posloader)

    for i, negdata in enumerate(tqdm(negloader, leave=False)):
        if 'CONT' in args.model:
            try:
                posdata = next(positerator)
            except StopIteration:
                positerator = iter(posloader)
                posdata = next(dataloader_iterator)
                # (img, value, episode) = data
                # img = img.reshape(img.shape[0]*img.shape[1], img.shape[2], img.shape[3], img.shape[4])
                # value = torch.flatten(value)
                # episode = torch.flatten(episode)
                # target = torch.cat((torch.unsqueeze(value, 1), torch.unsqueeze(episode, 1)), axis=1)

            if 'DUAL' in args.model:
                # for now we are testing with the same game
                # so the adapter learns an identity function
                negimg_reshape_val = negdata[0].to(device) / div_val
                posimg_reshape_val = posdata[0].to(device) / div_val
                negtar_reshape_val = negdata[1].to(device)
                postar_reshape_val = posdata[1].to(device)

                tquery = teachernet(negimg_reshape_val)
                equery = encodernet(posimg_reshape_val)
                tclass = negtar_reshape_val
                eclass = postar_reshape_val
                loss = loss_func(torch.cat([tquery, equery], axis=0), torch.cat([tclass, eclass], axis=0))



            # if its 4stack_cont_atari
            else:
                neg_reshape_val = negdata.to(device) / div_val
                pos_reshape_val = posdata.to(device) / div_val
                # batch_size, num_negative, embedding_size = 32, 48, 128
                # sample_batch_size -> number of positives/queries
                # train_batch_size -> number of negatives
                query_imgs, pos_imgs = torch.split(pos_reshape_val, 1, dim=1)

                if 'LSTM' in args.model:
                    encodernet.init_hs(batch_size=args.train_batch_size)
                    neg_reshape_val = torch.unsqueeze(neg_reshape_val, axis=2)
                    query_imgs = torch.reshape(query_imgs, (args.train_batch_size, args.maxseq, 1, 84, 84))
                    pos_imgs = torch.reshape(pos_imgs, (args.train_batch_size, args.maxseq, 1, 84, 84))
                else:
                    query_imgs = torch.squeeze(query_imgs)
                    pos_imgs = torch.squeeze(pos_imgs)

                query = encodernet(query_imgs)
                positives = encodernet(pos_imgs)
                negatives = encodernet(neg_reshape_val)

                if 'LSTM' in args.model:
                    query = query.reshape((-1, 512))
                    positives = positives.reshape((-1, 512))
                    negatives = negatives.reshape((-1, 512))

                # allocat classes for queries, positives and negatives
                posclasses = torch.arange(start=0, end=query.shape[0])
                negclasses = torch.arange(start=query.shape[0], end=query.shape[0] + negatives.shape[0])
                loss = loss_func(torch.cat([query, positives, negatives], axis=0),
                                 torch.cat([posclasses, posclasses, negclasses], axis=0))

        elif 'LSTM' in args.model:
            # CHEN
            (img, target, action) = negdata
            image_reshape_val = img.to(device) / div_val
            targ = target.to(device) / div_val
            action = action.to(device)

            z_gt, _, _ = encodernet.encode(targ)
            z_prev, _, _ = encodernet.encode(image_reshape_val)

            mask = random.sample(range(0, len(z_prev[0])), int(len(z_prev[0]) / 2))
            for z in z_prev:
                z[mask] = latent_cls[random.randint(0, 9)].to(device)
            
            encodernet.init_hs(z_prev[:, 0])

            #z_pred = encodernet(action, z_prev)
            #loss = loss_func(z_pred, z_gt)
            mus, sigmas, logpi = encodernet(action, z_prev)
            loss = loss_func(z_gt, mus, sigmas, logpi) / z_gt.shape[-1]

        else:
            (img, target) = negdata
            image_val = img.to(device) / div_val
            targ = target.to(device) / div_val

            recon_data, mu, logvar = encodernet(image_val)

            # in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative

            loss = loss_func(recon_data, targ, mu, logvar, args.kl_weight)

        # break
        loss_iter.append(loss.item())
        loss_log.append(loss.item())
        # from IPython import embed; embed()
        encodernet.zero_grad()
        loss.backward()
        optimizer.step()
    print("learning rate:", optimizer.param_groups[0]['lr'])

    print(np.mean(np.array(loss_iter)))
    if 'VAE' in args.model:
        auxval = args.kl_weight
    else:
        auxval = args.temperature

    if 'DUAL' in args.model:
        log_write(encodernet, 'a')

    # continue
    # Save a checkpoint with a specific filename
    torch.save({
        'epoch': epoch,
        'loss_log': loss_log,
        'model_state_dict': encodernet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()

    }, args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (args.arch).upper() + "_" + str(
        auxval) + "_" + str(args.sgamma) + "_" + str(args.train_batch_size) + "_" + str(args.sample_batch_size) + ".pt")

