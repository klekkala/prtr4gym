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
from models.atarimodels import SingleAtariModel
from torch.hub import load_state_dict_from_url
import os
import random
import numpy as np
import math
from ray.rllib.policy.policy import Policy
from IPython.display import clear_output
from ray.rllib.models import ModelCatalog
from PIL import Image
from tqdm import trange, tqdm
#from RES_VAE import VAE as VAE
from models.atari_vae import VAE, VAEBEV
from models.LSTM import LSTM as BEVLSTM
from models.atari_vae import Encoder, TEncoder
from dataclass import BaseDataset, FourStack, ThreeChannel, SingleChannel, SingleChannelLSTM, SingleAtari101, PosContFourStack, NegContFourStack
import utils
from arguments import get_args
args = get_args()


def initialize(is_train):

    root_dir = "/dev/shm/"
    curr_dir = os.getcwd()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device(args.gpu_id if use_cuda else "cpu")
    print(device)
    # %%


    #image_size = 84
    #if something looks wrong.. look at the transform line
    #OLD once changed on 5 june
    #transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    
    transform = T.Compose([T.ToTensor()])

    if args.model == "BEV_VAE_CARLA":
        trainset = CarlaBEV.CarlaBEV(root_dir= root_dir + args.expname, transform=transform)
        encodernet = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        div_val = 255.0
    
    #CHEN
    elif args.model == "BEV_LSTM_CARLA":
        trainset = SingleChannelLSTM.SingleChannelLSTM(root_dir= root_dir + args.expname, transform=transform)
        encodernet = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_E2E_0.01_256_64.pt"
        checkpoint = torch.load(model_path, map_location="cpu")
        encodernet.load_state_dict(checkpoint['model_state_dict'])
        encodernet = BEVLSTM(latent_size=512, action_size=2, hidden_size=512, batch_size=99, num_layers=1, vae=encodernet).to(device)
        
        div_val = 255.0


    elif args.model == "4STACK_VAE_ATARI":
        trainset = FourStack.FourStack(root_dir= root_dir + args.expname, transform=transform)
        encodernet = VAE(channel_in=4, ch=32, z=512).to(device)
        div_val = 255.0

    elif args.model == "3CHANRGB_VAE_ATARI101":
        trainset = Atari101.Atari101(root_dir='/lab/kiran/vae_d4rl/', transform=transform)
        encodernet = VAE(channel_in=3, ch=32, z=512).to(device)
        div_val = 1.0

    #incase you need a rgb model atari
    elif args.model == "1CHAN_VAE_ATARI101":
        trainset = SingleAtari101.SingleAtari101(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAE(channel_in=1, ch=32, z=512).to(device)
        div_val = 1.0

    #single channel vae used for notemp and lstm mode
    elif args.model == "1CHAN_VAE_ATARI":
        trainset = SingleChannel.SingleChannel(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAE(channel_in=1, ch=32, z=512).to(device)
        div_val = 255.0

    #our method.. note that 
    # cont4stack_atari: the dataloader will give 2 pairs of 4stack observations, their actions and their values
    # contlstm_atari: the dataloader will give 2 pairs of observation, action and value/reward arrays
    # contlstm_beogym: the dataloader will give 2 pairs of observation, action and value/reward arrays along with goal points for those.
    # BUT ULTIMATELY.. THE LOSS FUNCTION MUST GET EMBEDDINGS AND IF THEY ARE POSITIVE OR NEGATIVE

    elif args.model == "4STACK_CONT_ATARI":
        negset = NegContFourStack.NegContFourStack(root_dir= root_dir + args.expname, transform=transform)
        posset = PosContFourStack.PosContFourStack(root_dir=root_dir + args.expname, transform=transform, sample_next=args.sgamma)
        encodernet = Encoder(channel_in=4, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "DUAL_4STACK_CONT_ATARI":
        negset = NegContFourStack.NegContFourStack(root_dir= root_dir + args.expname, transform=transform)
        posset = PosContFourStack.PosContFourStack(root_dir=root_dir + args.expname, transform=transform, sample_next=args.sgamma)
        encodernet = Encoder(channel_in=4, ch=32, z=512).to(device)
        print(root_dir, args.expname)
        div_val = 255.0
        teachernet = TEncoder(channel_in=4, ch=16, z=512).to(device)

        #load teacher_encoder ckpt        
        ModelCatalog.register_custom_model("model", SingleAtariModel)
        load_ckpt = Policy.from_checkpoint("/lab/kiran/logs/rllib/atari/4stack/1.a_BeamRiderNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_4stack/23_07_04_18_11_34/checkpoint").get_weights()

        #transfer weights to prtr model
        model_state_dict = {}
        model_state_dict['encoder.1.weight'] = torch.from_numpy(load_ckpt['_convs.0._model.1.weight'])
        model_state_dict['encoder.1.bias'] = torch.from_numpy(load_ckpt['_convs.0._model.1.bias'])
        model_state_dict['encoder.4.weight'] = torch.from_numpy(load_ckpt['_convs.1._model.1.weight'])
        model_state_dict['encoder.4.bias'] = torch.from_numpy(load_ckpt['_convs.1._model.1.bias'])
        model_state_dict['encoder.6.weight'] = torch.from_numpy(load_ckpt['_convs.2._model.0.weight'])
        model_state_dict['encoder.6.bias'] = torch.from_numpy(load_ckpt['_convs.2._model.0.bias'])

        teachernet.eval()
        for param in teachernet.parameters():
            param.requires_grad = False
        teachernet.load_state_dict(model_state_dict)

        #initialize student model
        encodernet = TEncoder(channel_in=4, ch=16, z=512).to(device)
        print(root_dir, args.expname)
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

    if is_train and 'CONT' in args.model:
        negloader, posloader = utils.get_data_STL10(negset, args.train_batch_size, transform, posset, args.sample_batch_size)
    elif is_train:
        trainloader, _ = utils.get_data_STL10(trainset, args.train_batch_size, transform)
    else:
        trainloader, _ = utils.get_data_STL10(trainset, 20, transform)
        args.load_checkpoint = True

    # setup optimizer
    optimizer = optim.Adam(encodernet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # Loss function
    loss_log = []

    # %%

    # Create the results directory if it does note exist
    if not os.path.isdir(curr_dir + "/Results"):
        os.makedirs(curr_dir + "/Results")

    if args.load_checkpoint:
        
        if args.model_path == "":
            model_path = args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (args.arch).upper() + "_" + str(args.kl_weight) + "_" + str(args.sgamma) + "_" + str(args.train_batch_size) + "_" + str(args.sample_batch_size) + ".pt"
        else:
            model_path = args.save_dir + args.model_path
        checkpoint = torch.load(model_path, map_location="cpu")
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

    
    if 'DUAL' in args.model:
        return encodernet, teachernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
    elif 'CONT' in args.model:
        return encodernet, negloader, posloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
    else:
        return encodernet, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
