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
from IPython import embed
import os
import random
import numpy as np
import math
from IPython.display import clear_output

from PIL import Image
from tqdm import trange, tqdm
#from RES_VAE import VAE as VAE
from atari_vae import VAE as VAE

from dataclass import FourStackAtari, ThreeChannelAtari, NewFourStack, Atari101, SingleChannelAtari
import utils
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--arch",
    choices=["standard", "resnet"],
    default="e2e",
)
parser.add_argument(
    "--model",
    choices=["4STACK_ATARI", "RGB_ATARI101", "GREYSCALE_ATARI101", "3CHANNEL_ATARI", "GREY_ATARI", "CONT_ATARI"],
    default="e2e",
)
parser.add_argument(
    "--machine",
    choices=["iGpu", "iGpu8", "iGpu10", "iGpu14", "iGpu9"],
    default="e2e",
)
parser.add_argument(
    "--save_dir", type=str, default="/lab/kiran/ckpts/pretrained/atari/", help="pretrained results"
)
parser.add_argument(
    "--expname", type=str, default="all", help="pretrained results"
)
parser.add_argument(
    "--kl_weight", type=float, default=.01, help="pretrained results"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="pretrained results"
)
parser.add_argument(
    "--gpu_id", type=int, default=0, help="GPU ID"
)


# %%
args = parser.parse_args()
curr_dir = os.getcwd()
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device(args.gpu_id if use_cuda else "cpu")
print(device)
# %%

dir_map = {"4STACK_ATARI": "atari/4stack/", "GREY_ATARI": "atari/lstm/"}

if args.machine == "iGpu10":
    root_dir = "/home6/tmp/kiran/"
elif args.machine == "iGpu14":
    #root_dir = "/home3/tmp/kiran/"
    root_dir = "/dev/shm/"
elif args.machine == "iGpu":
    root_dir = "/dev/shm/"
else:
    root_dir = "/home/tmp/kiran/"


image_size = 84
#OLD once changed on 5 june
transform = T.Compose([T.Resize(image_size), T.ToTensor()])
#transform = T.Compose([T.ToTensor()])

if args.model == "4STACK_ATARI":
    trainset = FourStackAtari.FourStackAtari(root_dir= root_dir + dir_map[args.model] + args.expname, transform=transform)
    print(root_dir, dir_map[args.model])
    vae_net = VAE(channel_in=4, ch=32, z=512).to(device)
    div_val = 255.0

elif args.model == "GREYSCALE_ATARI101":
    trainset = Atari101.Atari101(root_dir='/lab/kiran/vae_d4rl/', transform=transform)
    vae_net = VAE(channel_in=3, ch=32, z=512).to(device)
    div_val = 1.0

elif args.model == "3CHANNEL_ATARI":
    trainset = ThreeChannelAtari.ThreeChannelAtari(root_dir='/home3/kiran/train/', transform=transform)
    vae_net = VAE(channel_in=3, ch=32, z=512).to(device)
    div_val = 1.0

elif args.model == "GREY_ATARI":
    #if args.expname == "all":
    #trainset = NewFourStack.NewFourStack(root_dir=root_dir + dir_map[args.model], transform=transform)
    #else:
    trainset = SingleChannelAtari.SingleChannelAtari(root_dir=root_dir + dir_map[args.model] + args.expname, transform=transform)
    vae_net = VAE(channel_in=1, ch=32, z=512).to(device)
    div_val = 255.0

elif args.model == "CONT_ATARI":
    trainset = ContAtari.ContAtari(root_dir='/home6/kiran/datasets/', transform=transform)
    vae_net = VAE(channel_in=4, ch=32, z=512).to(device)
    div_val = 255.0

else:
    raise("Not Implemented Error")


trainloader, _ = utils.get_data_STL10(trainset, None, transform, 5)

# setup optimizer
optimizer = optim.Adam(vae_net.parameters())

# Create the save directory if it does note exist
if not os.path.isdir(curr_dir + "/Results"):
    os.makedirs(curr_dir + "/Results")


checkpoint = torch.load(args.save_dir + args.model + "_" + args.expname.upper() + "_" + str(args.kl_weight) + "_" + str(args.batch_size) + ".pt", map_location="cpu")
print("Checkpoint loaded")
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
vae_net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint["epoch"]
print("Epoch starting at ", start_epoch)
loss_log = checkpoint["loss_log"]


# %%
vae_net.eval()
for i, (test_images, _) in enumerate(tqdm(trainloader, leave=False)):
    with torch.no_grad():
        #torch.unsqueeze(test_images, 1)
        test_images /= div_val
        test_images_gpu = test_images.to(device)
        recon_data, y, _ = vae_net(test_images_gpu.to(device))
        if args.model == "4STACK_ATARI":
            test_images = test_images.reshape(20, 1, 84, 84)
            recon_data = recon_data.reshape(20, 1, 84, 84)
        vutils.save_image(torch.cat((torch.sigmoid(recon_data).cpu(), test_images), 2),
                            "%s/%s/%s_%d.png" % (curr_dir, "Results", args.model, image_size))

        break
