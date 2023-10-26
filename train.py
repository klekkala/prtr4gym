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
#from evaluate import eval_adapter
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

game_ckpts = {
        'AirRaidNoFrameskip-v4': '/lab/kiran/logs/rllib/atari/4stack/1.a_AirRaidNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_01_30_05/checkpoint/',
        'CarnivalNoFrameskip-v4': '/lab/kiran/logs/rllib/atari/4stack/1.a_CarnivalNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_07_18_30_07/checkpoint/',
        'DemonAttackNoFrameskip-v4': '/lab/kiran/logs/rllib/atari/4stack/1.a_DemonAttackNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_12_34_09/checkpoint/',
        'NameThisGameNoFrameskip-v4': '/lab/kiran/logs/rllib/atari/4stack/1.a_NameThisGameNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_09_14_07/checkpoint/',
        'SpaceInvadersNoFrameskip-v4': '/lab/kiran/logs/rllib/atari/4stack/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_13_44_50/checkpoint/',
        }

game_trans = {
    'trained_4stack_airraid': 'AirRaidNoFrameskip-v4',
    'trained_4stack_carnival': 'CarnivalNoFrameskip-v4',
    'trained_4stack_demonattack': 'DemonAttackNoFrameskip-v4',
    'trained_4stack_namethisgame': 'NameThisGameNoFrameskip-v4',
    'trained_4stack_spaceinvaders': 'SpaceInvadersNoFrameskip-v4'
        }
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
            game_ckpts[game_trans[args.texpname]], game_ckpts[game_trans[args.expname]],
                game_trans[args.expname], encodernet)
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

elif 'TCN' in args.model or 'SOM' in args.model or 'VEP' in args.model or 'VIP' in args.model:
    encodernet, negloader, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(
        True)
elif 'CONT' in args.model:
    encodernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(
        True)
elif 'FPV' in args.model:
    fpvencoder, bevencoder, negloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(
        True)
else:
    encodernet, negloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)

# select which contrastive loss to train
if 'CONT' in args.model:
    # loss_func = InfoNCE(temperature=args.temperature, negative_mode='unpaired') # negative_mode='unpaired' is the default value
    loss_func = losses.ContrastiveLoss()

elif 'FPV' in args.model:
    #loss_func = utils.clip_loss
    loss_func = losses.ContrastiveLoss()
elif 'LSTM' in args.model and 'CARLA' in args.model:
    #loss_func = nn.MSELoss()
    loss_func = utils.gmm_loss
elif 'VIP' in args.model:
    loss_func = utils.vip_loss
elif 'TCN' in args.model or 'SOM' in args.model:
    loss_func = losses.ContrastiveLoss()

elif 'VEP' in args.model:
    loss_func = losses.ContrastiveLoss()
else:
    loss_func = utils.vae_loss

if 'DUAL' in args.model:
    log_write(encodernet, 'w')

if 'LSTM' in args.model and 'CARLA' in args.model:
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
    loss_iter = []

    # contrastive case: for i, (img_batch1, img_batch2, pair) in enumerate(tqdm(trainloader, leave=False)):
    # img_batch1, img_batch2 -> [B, T, H, W]

    if 'CONT' in args.model or 'VIP' in args.model or 'VEP' in args.model or 'SOM' in args.model:
        encodernet.train()
        negiterator = iter(negloader)
    if 'BEV_VAE' in args.model:
        encodernet.train()
        anchors = []
        for i in range(11):
            img = cv2.imread("manual_label/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=(0, 1))
            img_val = torch.tensor(img).to(device) / 255.0
            anchors.append(img_val)
    elif 'FPV' in args.model:
        fpvencoder.train()
        bevencoder.train()
    else:
        encodernet.train()

    for i, traindata in enumerate(tqdm(trainloader, leave=False)):


        if 'CONT' in args.model:
            try:
                posdata = next(positerator)
            except StopIteration:
                positerator = iter(posloader)
                posdata = next(positerator)
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

                #ONLY FOR CONT!!!
                if '1CHANLSTM' in args.model:
                    encodernet.init_hs(args.train_batch_size)

                    neg_reshape_val = torch.unsqueeze(torch.squeeze(neg_reshape_val), axis=1)
                    query_imgs = torch.unsqueeze(torch.squeeze(query_imgs), axis=1)
                    pos_imgs = torch.unsqueeze(torch.squeeze(pos_imgs), axis=1)

                    #maxseq = max(neg_reshape_val.shape[1], query_imgs.shape[1])
                    #zs = np.zeros((self.max_seq_length - inputtraj.shape[0],) + inputtraj.shape[1:]).astype(np.float32)

                    #neg_reshape_val = torch.unsqueeze(neg_reshape_val, axis=2)
                    #query_imgs = torch.reshape(query_imgs, (args.train_batch_size, neg_reshape_val.shape[1], 1, 84, 84))
                    #pos_imgs = torch.reshape(pos_imgs, (args.train_batch_size, pos_imgs.shape[1], 1, 84, 84))

                elif '3CHANLSTM' in args.model:
                    encodernet.init_hs(args.train_batch_size)
                    neg_reshape_val = torch.squeeze(neg_reshape_val)
                    query_imgs = torch.squeeze(query_imgs)
                    pos_imgs = torch.squeeze(pos_imgs)


                elif '1CHAN_CONT' in args.model:
                    query_imgs = torch.unsqueeze(torch.squeeze(query_imgs), axis=1)
                    pos_imgs = torch.unsqueeze(torch.squeeze(pos_imgs), axis=1)
                    
                elif '3CHAN_CONT' in args.model:
                    query_imgs = torch.squeeze(query_imgs)
                    pos_imgs = torch.squeeze(pos_imgs)

                else:
                    query_imgs = torch.squeeze(query_imgs)
                    pos_imgs = torch.squeeze(pos_imgs)

                query = encodernet(query_imgs)
                positives = encodernet(pos_imgs)
                negatives = encodernet(neg_reshape_val)

                #EMBED HERE!
                #embed()

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
            if len(action[0]) <= 1:
                continue
            ids, sim, image_embed = encodernet.encoder(img[0])
            source = encodernet.encoder.anchors[ids]
            source = torch.tensor(source)

            image_reshape_val = source.to(device) / div_val
            targ = target.to(device) / div_val
            action = action.to(device)
            z_gt, _, _ = encodernet.encode(targ)
            z_prev, _, _ = encodernet.encode(image_reshape_val.unsqueeze(0).unsqueeze(2))

            #mask = random.sample(range(0, len(z_prev[0])), int(len(z_prev[0]) / 2))
            #for z in z_prev:
            #    z[mask] = latent_cls[random.randint(0, 9)].to(device)
            
            encodernet.init_hs(len(img))

            #z_pred = encodernet(action, z_prev)
            #loss = loss_func(z_pred, z_gt)
            mus, sigmas, logpi = encodernet(action, z_prev)
            loss = loss_func(z_gt, mus, sigmas, logpi) / z_gt.shape[-1]

        elif "FPV_RECONBEV_CARLA" in args.model:
            (img, target) = negdata
            image_val = img.to(device) / div_val
            targ = target.to(device) / div_val

            image_embed = fpvencoder(image_val)
            #_, targ_embed_mu, targ_embed_logvar = bevencoder(targ)
            #targ_embed = targ_embed_mu
            #torch.concat((targ_embed_mu, targ_embed_logvar), axis=-1)
            #embedclasses = torch.arange(start=0, end=img.shape[0])

            # in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
            #loss = loss_func(torch.cat([image_embed, targ_embed], axis=0), torch.cat([embedclasses, embedclasses], axis=0))
            #loss = loss_func(image_embed, targ_embed, args.temperature, embedclasses)
            
            # reconstruct from fpv embedding
            z = bevencoder.reparameterize(image_embed[:, :32], image_embed[:, 32:])
            recon_data = bevencoder.recon(z)
            #logvar = torch.zeros(image_embed.shape).to(device)

            loss = utils.vae_loss(recon_data, targ, image_embed[:, :32], image_embed[:, 32:], args.kl_weight)

            # regression
#            loss = F.mse_loss(image_embed, targ_embed)


        elif 'FPV' in args.model:
            (img, target) = negdata
            image_val = img.to(device) / div_val
            targ = target.to(device) / div_val

            image_embed = fpvencoder(image_val)
            _, targ_embed_mu, targ_embed_logvar = bevencoder(targ)
            targ_embed = targ_embed_mu
            #torch.concat((targ_embed_mu, targ_embed_logvar), axis=-1)
            embedclasses = torch.arange(start=0, end=img.shape[0])

            # in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
            #loss = loss_func(torch.cat([image_embed, targ_embed], axis=0), torch.cat([embedclasses, embedclasses], axis=0))
            loss = loss_func(image_embed, targ_embed, args.temperature, embedclasses)
            
            # regression
#            loss = F.mse_loss(image_embed, targ_embed)

        elif 'TCN' in args.model:

            #try:
            #    posdata = next(positerator)
            #except StopIteration:
            #    positerator = iter(posloader)
            #    posdata = next(positerator)
            
            
            #don't use this yet
            neg_reshape_val = negdata.to(device) / div_val
            #pos_reshape_val = posdata.to(device) / div_val

            query_imgs, pos_imgs, neg_imgs = torch.split(neg_reshape_val, 1, dim=1)

            query = encodernet(query_imgs.reshape(query_imgs.shape[0], 1, 84, 84))
            positives = encodernet(pos_imgs.reshape(pos_imgs.shape[0], 1, 84, 84))
            negatives = encodernet(neg_imgs.reshape(neg_reshape_val.shape[0], 1, 84, 84))
            
            # allocat classes for queries, positives and negatives
            posclasses = torch.arange(start=0, end=query.shape[0])
            negclasses = torch.arange(start=query.shape[0], end=query.shape[0] + negatives.shape[0])
            loss = loss_func(torch.cat([query, positives, negatives], axis=0),
                                torch.cat([posclasses, posclasses, negclasses], axis=0))

        elif 'SOM' in args.model:

            try:
                negdata = next(negiterator)
            except StopIteration:
                negiterator = iter(negloader)
                negdata = next(negiterator)
            
            
            #don't use this yet
            neg_reshape_val = negdata.to(device) / div_val
            pos_reshape_val = posdata.to(device) / div_val
            #embed()
            query_imgs, pos_imgs, neg_imgs = torch.split(pos_reshape_val, 1, dim=1)

            if args.sample_batch_size == args.train_batch_size:
                neg_imgs = neg_reshape_val

            query = encodernet(query_imgs.reshape(query_imgs.shape[0], 1, 84, 84))
            positives = encodernet(pos_imgs.reshape(pos_imgs.shape[0], 1, 84, 84))
            negatives = encodernet(neg_imgs.reshape(neg_imgs.shape[0], 1, 84, 84))
            
            # allocat classes for queries, positives and negatives
            posclasses = torch.arange(start=0, end=query.shape[0])
            negclasses = torch.arange(start=query.shape[0], end=query.shape[0] + negatives.shape[0])
            loss = loss_func(torch.cat([query, positives, negatives], axis=0),
                                torch.cat([posclasses, posclasses, negclasses], axis=0))


        elif 'VEP' in args.model:

            try:
                negdata = next(negiterator)
            except StopIteration:
                negiterator = iter(negloader)
                negdata = next(negiterator)
            
            
            #don't use this yet
            #neg_reshape_val = negdata.to(device) / div_val
            pos_reshape_val = traindata.to(device) / div_val

            
            #each of the items on the left is of size Bx2
            query_imgs, pos_imgs, neg_imgs, goal_imgs = torch.split(pos_reshape_val, 2, dim=1)

            # allocat classes for queries, positives, negatives and goals
            posclasses = torch.arange(start=0, end=query_imgs.shape[0]).repeat(2, 1).T
            goalclasses = torch.arange(start=query_imgs.shape[0], end=query_imgs.shape[0] + goal_imgs.shape[0]).repeat(2,1).T
            negclasses = torch.arange(start=query_imgs.shape[0]+goal_imgs.shape[0], end=query_imgs.shape[0] + goal_imgs.shape[0] + neg_imgs.shape[0]*2).reshape(neg_imgs.shape[0], 2)

            query_imgs = query_imgs.reshape(query_imgs.shape[0]*query_imgs.shape[1], 1, 84, 84)
            pos_imgs = pos_imgs.reshape(pos_imgs.shape[0]*pos_imgs.shape[1], 1, 84, 84)
            neg_reshape_val = neg_imgs.reshape(neg_imgs.shape[0]*neg_imgs.shape[1], 1, 84, 84)
            goal_imgs = goal_imgs.reshape(goal_imgs.shape[0]*goal_imgs.shape[1], 1, 84, 84)
            

            query = encodernet(query_imgs)
            positives = encodernet(pos_imgs)
            negatives = encodernet(neg_reshape_val)
           

            if 'NVEP' in args.model:
                loss = loss_func(torch.cat([query, positives, negatives], axis=0),
                    torch.cat([posclasses.flatten(), posclasses.flatten(), negclasses.flatten()], axis=0))
            else:
                goals = encodernet(goal_imgs)
                loss = loss_func(torch.cat([query, positives, negatives, goals], axis=0),
                    torch.cat([posclasses.flatten(), posclasses.flatten(), negclasses.flatten(), goalclasses.flatten()], axis=0))
            
            

        elif 'VIP' in args.model:

            try:
                posdata = next(positerator)
            except StopIteration:
                positerator = iter(posloader)
                posdata = next(positerator)
            _, add_mid_imgs, add_midplus_imgs, _ = torch.split(posdata, 1, dim=1)
            



            start_imgs, mid_imgs, midplus_imgs, end_imgs = torch.split(negdata, 1, dim=1)
            
            start_imgs = torch.squeeze(start_imgs.to(device)/div_val)
            mid_imgs = torch.squeeze(mid_imgs.to(device)/div_val)
            add_mid_imgs = torch.squeeze(add_mid_imgs.to(device)/div_val)
            midplus_imgs = torch.squeeze(midplus_imgs.to(device)/div_val)
            add_midplus_imgs = torch.squeeze(add_midplus_imgs.to(device)/div_val)
            end_imgs = torch.squeeze(end_imgs.to(device)/div_val)
            
            if len(start_imgs.shape) == 3:
                start_imgs = torch.unsqueeze(start_imgs, axis=1)
                mid_imgs = torch.unsqueeze(mid_imgs, axis=1)
                add_mid_imgs = torch.unsqueeze(add_mid_imgs, axis=1)
                midplus_imgs = torch.unsqueeze(midplus_imgs, axis=1)
                add_midplus_imgs = torch.unsqueeze(add_midplus_imgs, axis=1)
                end_imgs = torch.unsqueeze(end_imgs, axis=1)

            start_embed = encodernet(start_imgs)
            mid_embed = encodernet(mid_imgs)
            add_mid_embed = encodernet(mid_imgs)
            midplus_embed = encodernet(midplus_imgs)
            add_midplus_embed = encodernet(midplus_imgs)
            end_embed = encodernet(end_imgs)

            loss = loss_func(start_embed, mid_embed, add_mid_embed, midplus_embed, add_midplus_embed, end_embed)
        
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

        #print(np.mean(np.array(loss_iter)))
        # from IPython import embed; embed()
        if 'FPV' in args.model:
            fpvencoder.zero_grad()
            bevencoder.zero_grad()
        else:
            encodernet.zero_grad()
        loss.backward()
        optimizer.step()
        if 'VAE' in args.model:
            auxval = args.kl_weight
        else:
            auxval = args.temperature

    print("learning rate:", optimizer.param_groups[0]['lr'])

    print(np.mean(np.array(loss_iter)))
    if 'VAE' in args.model:
        auxval = str(args.kl_weight)
    elif 'VIP' in args.model:
        auxval = str(args.max_len) + '_' + str(args.min_len)
    elif 'VEP' in args.model:
        auxval = str(args.max_len) + '_' + str(args.temperature) + '_' + str(args.dthresh) + '_' + str(args.negtype)
    elif 'TCN':
        auxval = str(args.max_len)
    elif 'SOM':
        auxval = str(args.sgamma)
    else:
        auxval = str(args.temperature)
    

    if 'DUAL' in args.model:
        log_write(encodernet, 'a')

    # continue
    # Save a checkpoint with a specific filename

    #if epoch % (args.nepoch/20) == 0:
    #if epoch % (args.nepoch/4) == 0:
    if True:
        print("saving checkpoint")
        if 'FPV' in args.model:
            save_dict = {
                'epoch': epoch,
                'loss_log': loss_log,
                'fpv_state_dict': fpvencoder.state_dict(),
                'bev_state_dict': bevencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch,
                'loss_log': loss_log,
                'model_state_dict': encodernet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }    
        torch.save(save_dict, args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (args.arch).upper() + "_" + auxval + "_" + str(args.train_batch_size) + "_" + str(args.sample_batch_size) + "_" + str(args.lr) + "_" + str(epoch) + ".pt")
