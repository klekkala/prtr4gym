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
import os
import random
import numpy as np
import math
from IPython.display import clear_output
from PIL import Image
from tqdm import trange, tqdm
#from RES_VAE import VAE as VAE
from models.atari_vae import VAE as VAE
from dataclass import BaseDataset, FourStack, ThreeChannel, Atari101, SingleChannel
import utils
from arguments import get_args
args = get_args()


def initialize(is_train):

    
    curr_dir = os.getcwd()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device(args.gpu_id if use_cuda else "cpu")
    print(device)
    # %%

    count_map = {"medium_1chan_beamrider": 1000000, "all_1chan_beamrider": 3000000, "medium_1chan_train": 5000000, "medium_1chan_test": 4000000, "medium_1chan_beamrider": 4000000, "medium_4stack_beamrider": 1000000, "medium_4stack_train": 5000000, "medium_4stack_test": 4000000}

    if args.machine == "iGpu10":
        root_dir = "/home6/tmp/kiran/"
    elif args.machine == "iGpu14":
        #root_dir = "/home3/tmp/kiran/"
        root_dir = "/dev/shm/"
    elif args.machine == "iGpu":
        root_dir = "/dev/shm/"
    elif args.machine == "iGpu24":
        root_dir = "/dev/shm/"
    elif args.machine == "iGpu15":
        root_dir = "/dev/shm/"
    else:
        root_dir = "/home/tmp/kiran/"


    image_size = 84
    #OLD once changed on 5 june
    transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    #transform = T.Compose([T.ToTensor()])

    #stacking four frames. deepmind style
    if args.model == "4STACK_VAE_ATARI":
        trainset = FourStack.FourStack(root_dir= root_dir + args.expname, max_len=count_map[args.expname], transform=transform)
        encodernet = VAE(channel_in=4, ch=32, z=512).to(device)
        div_val = 255.0

    elif args.model == "3CHANRGB_VAE_ATARI101":
        trainset = Atari101.Atari101(root_dir='/lab/kiran/vae_d4rl/', max_len=count_map[args.expname], transform=transform)
        encodernet = VAE(channel_in=3, ch=32, z=512).to(device)
        div_val = 1.0

    #incase you need a rgb model atari
    elif args.model == "1CHAN_VAE_ATARI101":
        trainset = ThreeChannel.ThreeChannel(root_dir='/home3/kiran/train/', max_len=count_map[args.expname], transform=transform)
        encodernet = VAE(channel_in=3, ch=32, z=512).to(device)
        div_val = 1.0

    #single channel vae used for notemp and lstm mode
    elif args.model == "1CHAN_VAE_ATARI":
        trainset = SingleChannel.SingleChannel(root_dir=root_dir + args.expname, max_len=count_map[args.expname], transform=transform)
        encodernet = VAE(channel_in=1, ch=32, z=512).to(device)
        div_val = 255.0

    #our method.. note that 
    # cont4stack_atari: the dataloader will give 2 pairs of 4stack observations, their actions and their values
    # contlstm_atari: the dataloader will give 2 pairs of observation, action and value/reward arrays
    # contlstm_beogym: the dataloader will give 2 pairs of observation, action and value/reward arrays along with goal points for those.
    # BUT ULTIMATELY.. THE LOSS FUNCTION MUST GET EMBEDDINGS AND IF THEY ARE POSITIVE OR NEGATIVE

    elif args.model == "4STACK_CONT_ATARI":
        trainset = ContAtari.ContAtari(root_dir='/home6/kiran/datasets/', transform=transform)
        encodernet = VAE(channel_in=4, ch=32, z=512).to(device)
        #encodernet = Encoder()
        div_val = 255.0

    elif args.model == "CONTLSTM_ATARI":
        trainset = ContLstmAtari.ContLstmAtari(root_dir='/home6/kiran/datasets/', transform=transform)
        encodernet = VAE(channel_in=4, ch=32, z=512).to(device)
        #encodernet = Encoder()
        div_val = 255.0

    elif args.model == "CONTLSTM_BEOGYM":
        trainset = ContLstmBeogym.ContLstmBeogym(root_dir='/home6/kiran/datasets/', transform=transform)
        encodernet = VAE(channel_in=4, ch=32, z=512).to(device)
        #encodernet = Encoder()
        div_val = 255.0


    else:
        raise("Not Implemented Error")

    # %%
    # get a test image batch from the testloader to visualise the reconstruction quality
    #dataiter = iter(testloader)
    #test_images, _ = dataiter.next()

    if is_train:
        trainloader, _ = utils.get_data_STL10(trainset, None, transform, args.batch_size)
    else:
        trainloader, _ = utils.get_data_STL10(trainset, None, transform, 5)


    # setup optimizer
    optimizer = optim.Adam(encodernet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # Loss function
    loss_log = []

    # %%

    # Create the results directory if it does note exist
    if not os.path.isdir(curr_dir + "/Results"):
        os.makedirs(curr_dir + "/Results")

    if args.load_checkpoint:
        checkpoint = torch.load(args.save_dir + args.model + "_" + args.expname.upper() + ".pt", map_location="cpu")
        print("Checkpoint loaded")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        encodernet.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint["epoch"]
        print("Epoch starting at ", start_epoch)
        loss_log = checkpoint["loss_log"]
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        #if os.path.isfile(args.save_dir + args.model + ".pt"):
        #    raise ValueError("Warning Checkpoint exists. Overwriting")
        #else:
        #    print("Starting from scratch")
        start_epoch = 0

        
    return encodernet, trainloader, div_val, start_epoch, loss_log, optimizer, device