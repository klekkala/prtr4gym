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
from IPython import embed
from torch.hub import load_state_dict_from_url
import os
import random
import numpy as np
import math
#from ray.rllib.policy.policy import Policy
from IPython.display import clear_output
#from ray.rllib.models import ModelCatalog
from PIL import Image
from tqdm import trange, tqdm
from models.RES_VAE import TEncoder as ResEncoder
from models.atari_vae import VAE, VAEBEV
from models.LSTM import StateLSTM
from models.LSTM import MDLSTM as BEVLSTM
from models.resnet import ResNet
from models.BEVEncoder import BEVEncoder
from models.atari_vae import Encoder, TEncoder
from dataclass import BaseDataset, CarlaBEV, ThreeChannel, SingleChannel, SingleChannelLSTM, SingleAtari101, NegContSingleChan, PosContLSTM, NegContLSTM, TCNContSingleChan, PosContThreeLSTM, NegContThreeLSTM, CarlaFPVBEV, NegContThreeChan, PosContThreeChan, VEPContSingleChan, VIPContSingleChan, AtariVIPDataLoad, TCNContSingleChan, SOMContSingleChan
import utils
from arguments import get_args

args = get_args()


def initialize(is_train):
    if 'CARLA' in args.model:
        root_dir = "/home/tmp/kiran/"
    elif 'BEV_LSTM' in args.model:
        root_dir = "/home/tmp/kiran/"
    elif 'BEOGYM' in args.model or 'ATARI' in args.model:
        root_dir = "/dev/shm/"
    else:
        root_dir = "/dev/shm/"
    curr_dir = os.getcwd()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device(args.gpu_id if use_cuda else "cpu")
    print(device)
    # %%

    # image_size = 84
    # if something looks wrong.. look at the transform line
    # OLD once changed on 5 june
    # transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    transform = T.Compose([T.ToTensor()])

    if args.model == "BEV_VAE_CARLA":
        trainset = CarlaBEV.CarlaBEV(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        div_val = 255.0

    # CHEN
    elif args.model == "BEV_LSTM_CARLA":
        trainset = SingleChannelLSTM.SingleChannelLSTM(root_dir=root_dir + args.expname, transform=transform)
        vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        vae.load_state_dict(vae_ckpt['model_state_dict'])
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        
        encodernet = BEVLSTM(latent_size=32, action_size=2, hidden_size=256, gaussian_size=5,
                             num_layers=1, vae=vae).to(device)
        div_val = 255.0


    elif args.model == "4STACK_VAE_ATARI":
        trainset = FourStack.FourStack(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAE(channel_in=4, ch=32, z=512).to(device)
        div_val = 255.0

    elif args.model == "3CHANRGB_VAE_ATARI101":
        trainset = Atari101.Atari101(root_dir='/lab/kiran/vae_d4rl/', transform=transform)
        encodernet = VAE(channel_in=3, ch=32, z=512).to(device)
        div_val = 1.0

    # incase you need a rgb model atari
    elif args.model == "1CHAN_VAE_ATARI101":
        trainset = SingleAtari101.SingleAtari101(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAE(channel_in=1, ch=32, z=512).to(device)
        div_val = 1.0

    # single channel vae used for notemp and lstm mode
    elif args.model == "1CHAN_VAE_ATARI":
        trainset = SingleChannel.SingleChannel(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAE(channel_in=1, ch=32, z=512).to(device)
        div_val = 255.0

    # our method.. note that
    # cont4stack_atari: the dataloader will give 2 pairs of 4stack observations, their actions and their values
    # contlstm_atari: the dataloader will give 2 pairs of observation, action and value/reward arrays
    # contlstm_beogym: the dataloader will give 2 pairs of observation, action and value/reward arrays along with goal points for those.
    # BUT ULTIMATELY.. THE LOSS FUNCTION MUST GET EMBEDDINGS AND IF THEY ARE POSITIVE OR NEGATIVE

    elif args.model == "1CHAN_SOM_ATARI":
        negset = NegContSingleChan.NegContSingleChan(root_dir=root_dir + args.expname, transform=transform)
        trainset = SOMContSingleChan.SOMContSingleChan(root_dir=root_dir + args.expname, transform=transform, sample_next=args.sgamma)
        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=1, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=1, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "1CHAN_TCN_ATARI":
        negset = NegContSingleChan.NegContSingleChan(root_dir=root_dir + args.expname, transform=transform)
        trainset = TCNContSingleChan.TCNContSingleChan(root_dir=root_dir + args.expname, transform=transform, pos_distance=args.max_len)
        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=1, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=1, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "3CHAN_CONT_BEOGYM":
        negset = NegContThreeChan.NegContThreeChan(root_dir=root_dir + args.expname, transform=transform, goal=True)
        posset = PosContThreeChan.PosContThreeChan(root_dir=root_dir + args.expname, transform=transform, sample_next=args.sgamma, value=False, episode=True, goal=True)

        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=3, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=3, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "1CHAN_VIP_ATARI":
        trainset = AtariVIPDataLoad.AtariVIPDataLoad(root_dir=root_dir + args.expname, transform=transform, max_len=args.max_len, min_len=args.min_len, goal=False)
        negset = AtariVIPDataLoad.AtariVIPDataLoad(root_dir=root_dir + args.expname, transform=transform, max_len=args.max_len, min_len=args.min_len, goal=False)

        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=1, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=1, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "1CHAN_VEP_ATARI" or "1CHAN_NVEP_ATARI":
        trainset = VEPContSingleChan.VEPContSingleChan(root_dir=root_dir + args.expname, transform=transform, threshold=args.temperature, max_len = args.max_len, dthresh = args.dthresh, negtype = args.negtype, goal=False)
        negset = NegContSingleChan.NegContSingleChan(root_dir=root_dir + args.expname, transform=transform, goal=False)

        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=1, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=1, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "3CHAN_VIP_BEOGYM":
        trainset = VIPDataLoad.VIPDataLoad(root_dir=root_dir + args.expname, transform=transform, goal=True)

        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=3, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=3, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "4STACK_VIP_ATARI":
        trainset = FourStackAtariVIPDataLoad.FourStackAtariVIPDataLoad(root_dir=root_dir + args.expname, transform=transform, goal=False)

        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encodernet = TEncoder(channel_in=4, ch=64, z=512).to(device)
        else:
            encodernet = TEncoder(channel_in=4, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0


    elif args.model == "FPV_BEV_CARLA":
        trainset = CarlaFPVBEV.CarlaFPVBEV(root_dir=root_dir + args.expname, transform=transform)
        #this will give me a tuple: (fpv_batch, bev_batch), each of the batches are of size batch_size
        #CHEN
        #resnet150
        fpvencoder = ResNet(32).to(device)
        
        #vaeencoder
        bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        bevencoder.load_state_dict(vae_ckpt['model_state_dict'])
        bevencoder.eval()
        for param in bevencoder.parameters():
            param.requires_grad = False

        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "FPV_RECONBEV_CARLA":
        trainset = CarlaFPVBEV.CarlaFPVBEV(root_dir=root_dir + args.expname, transform=transform)
        #this will give me a tuple: (fpv_batch, bev_batch), each of the batches are of size batch_size
        #CHEN
        #resnet150
        fpvencoder = ResNet(64).to(device)
        
        #vae decoder
        bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        print(root_dir, args.expname)
        div_val = 255.0



    elif args.model == "1CHANLSTM_CONT_ATARI":
        negset = NegContLSTM.NegContLSTM(root_dir=root_dir + args.expname, transform=transform,
                                         max_seq_length=args.maxseq)
        posset = PosContLSTM.PosContLSTM(root_dir=root_dir + args.expname, transform=transform, sample_next=args.sgamma,
                                         max_seq_length=args.maxseq)
        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encoder = TEncoder(channel_in=1, ch=64, z=512).to(device)
        else:
            encoder = TEncoder(channel_in=1, ch=32, z=512).to(device)
        encodernet = StateLSTM(latent_size=512, hidden_size=512, num_layers=1,
                               encoder=encoder)

        print(root_dir, args.expname)
        div_val = 255.0


    elif args.model == "DUAL_4STACK_CONT_ATARI":
        negset = ContFourStack.ContFourStack(root_dir=root_dir + args.texpname, transform=transform)
        # posset = ContFourStack.ContFourStack(root_dir=root_dir + args.expname, transform=transform, max_len=50000)
        posset = ContFourStack.ContFourStack(root_dir=root_dir + args.expname, transform=transform)
        print(root_dir, args.expname)
        div_val = 255.0
        print("hahaha")
        teachernet = TEncoder(channel_in=4, ch=64, z=512).to(device)

        # load teacher_encoder ckpt
        # ModelCatalog.register_custom_model("model", SingleAtariModel)
        # load_ckpt = Policy.from_checkpoint("/lab/kiran/logs/rllib/atari/4stack/1.a_BeamRiderNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_4stack/23_07_04_18_11_34/checkpoint").get_weights()

        # transfer weights to prtr model
        if args.tmodel_path == "":
            model_path = args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (
                args.arch).upper() + "_" + str(args.kl_weight) + "_" + str(args.sgamma) + "_" + str(
                args.train_batch_size) + "_" + str(args.sample_batch_size) + ".pt"
        else:
            model_path = args.save_dir + args.tmodel_path
        checkpoint = torch.load(model_path, map_location="cpu")
        teachernet.load_state_dict(checkpoint['model_state_dict'])

        # freeze teachers model
        teachernet.eval()
        for param in teachernet.parameters():
            param.requires_grad = False

        print("bababa")
        # initialize student model
        encodernet = Encoder(channel_in=4, ch=64, z=512).to(device)
        checkpoint['model_state_dict']['adapter.weight'] = torch.from_numpy(np.ones((512, 512, 1, 1)))
        checkpoint['model_state_dict']['adapter.bias'] = torch.from_numpy(np.zeros(512))
        encodernet.load_state_dict(checkpoint['model_state_dict'])
        print("ggggggg")
        print(root_dir, args.expname)

        encodernet.encoder.eval()
        for param in encodernet.encoder.parameters():
            param.requires_grad = False

        encodernet.conv_mu.eval()
        for param in encodernet.conv_mu.parameters():
            param.requires_grad = False
        div_val = 255.0

    elif args.model == "3CHANLSTM_CONT_BEOGYM":
        negset = NegContThreeLSTM.NegContThreeLSTM(root_dir=root_dir + args.expname, transform=transform,
                                         max_seq_length=args.maxseq)
        posset = PosContThreeLSTM.PosContThreeLSTM(root_dir=root_dir + args.expname, transform=transform, sample_next=args.sgamma,
                                         max_seq_length=args.maxseq)
        if args.arch == 'resnet':
            print("using resnet")
            # encodernet = ResEncoder(channel_in=4, ch=64, z=512).to(device)
            encoder = TEncoder(channel_in=3, ch=64, z=512).to(device)
        else:
            encoder = TEncoder(channel_in=3, ch=32, z=512).to(device)
        encodernet = StateLSTM(latent_size=512, hidden_size=512, num_layers=1,
                               encoder=encoder)

        print(root_dir, args.expname)
        div_val = 255.0


    else:
        raise ("Not Implemented Error")

    # %%
    # get a test image batch from the testloader to visualise the reconstruction quality
    # dataiter = iter(testloader)
    # test_images, _ = dataiter.next()

    if is_train and 'CONT' in args.model:
        negloader, posloader = utils.get_data_STL10(negset, args.train_batch_size, transform, posset, args.sample_batch_size)
    
    elif is_train and 'VEP' in args.model or 'VIP' in args.model or 'TCN' in args.model or 'SOM' in args.model:
        if args.sample_batch_size > 0:
            negloader, trainloader = utils.get_data_STL10(negset, args.sample_batch_size, transform, trainset, args.sample_batch_size)
        else:
            negloader, trainloader = utils.get_data_STL10(negset, args.train_batch_size, transform, trainset, args.train_batch_size)


    elif is_train:
        trainloader, _ = utils.get_data_STL10(trainset, args.train_batch_size, transform)
    elif is_train == False and 'LSTM' in args.model:
        trainloader, _ = utils.get_data_STL10(trainset, 1, transform)
        args.load_checkpoint = True
    else:
        trainloader, _ = utils.get_data_STL10(trainset, 20, transform)
        args.load_checkpoint = True
    
    
    # setup optimizer
    if 'FPV' in args.model:
        optimizer = optim.Adam(list(fpvencoder.parameters()) + list(bevencoder.parameters()), lr=args.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.Adam(encodernet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # Loss function
    loss_log = []

    # %%

    # Create the results directory if it does note exist
    if not os.path.isdir(curr_dir + "/Results"):
        os.makedirs(curr_dir + "/Results")

    if args.load_checkpoint:
        if 'VAE' in args.model:
            auxval = args.kl_weight
        elif 'VIP' in args.model or 'TCN':
            auxval = args.max_len
        else:
            auxval = args.temperature
        if args.model_path == "":
            #this one will throw a bug!!!!
            model_path = args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (
                args.arch).upper() + "_" + str(auxval) + "_" + str(args.sgamma) + "_" + str(
                args.train_batch_size) + "_" + str(args.sample_batch_size) + "_" + str(args.lr) + "_" + str(epoch) + ".pt"
        else:
            model_path = args.save_dir + args.model_path
        
        print(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        print("Checkpoint loaded")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        start_epoch = checkpoint["epoch"]
        print("Epoch starting at ", start_epoch)
        loss_log = checkpoint["loss_log"]
        
        if 'FPV' in args.model:
            fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
            bevencoder.load_state_dict(checkpoint['bev_state_dict'])
        else:
            encodernet.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        # if os.path.isfile(args.save_dir + args.model + ".pt"):
        #    raise ValueError("Warning Checkpoint exists. Overwriting")
        # else:
        #    print("Starting from scratch")
        start_epoch = 0

    if 'DUAL' in args.model:
        return encodernet, teachernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
    elif 'CONT' in args.model or 'TCN' in args.model or 'VEP' in args.model or 'SOM' in args.model or 'VIP' in args.model:
        return encodernet, negloader, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
    elif 'FPV_BEV' in args.model or "FPV_RECONBEV_CARLA" in args.model:
        return fpvencoder, bevencoder, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir    
    else:
        return encodernet, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
