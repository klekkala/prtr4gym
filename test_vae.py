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
import math
from IPython.display import clear_output
from PIL import Image
from tqdm import trange, tqdm
import utils
import initial

args = get_args()

encodernet, testloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(False)



# %%
encodernet.eval()
for i, (test_images, _) in enumerate(tqdm(testloader, leave=False)):
    with torch.no_grad():
        print(i)
        #torch.unsqueeze(test_images, 1)
        test_images /= div_val
        test_images_gpu = test_images.to(device)
        recon_data, y, _ = encodernet(test_images_gpu.to(device))
        if "4STACK" in args.model:
            test_images = test_images.reshape(20, 1, 84, 84)
            recon_data = recon_data.reshape(20, 1, 84, 84)
        vutils.save_image(torch.cat((torch.sigmoid(recon_data).cpu(), test_images), 2),
                            "%s/%s/%s_%s.png" % (curr_dir, "Results", args.model, args.expname.upper() + "_" + str(args.kl_weight)))

        break
