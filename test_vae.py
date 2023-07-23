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
for i, testdata in enumerate(tqdm(testloader, leave=False)):
    with torch.no_grad():
        print(i)
        #torch.unsqueeze(test_images, 1)
        if 'LSTM' in args.model:
            (img, target, action) = testdata
            image_reshape_val = img.to(device)/div_val
            targ = target.to(device)/div_val
            action = action.to(device)

            encodernet.init_hs()
            z_gt, _, _ = encodernet.encode(targ)
            z_prev, _, _ = encodernet.encode(image_reshape_val)
            z_pred = encodernet(action, z_prev)
            test_images = encodernet.decode(z_gt).reshape((-1,) + img.shape[2:])
            #test_images = targ.reshape((-1,) + img.shape[2:])
            recon_data = encodernet.decode(z_pred).reshape((-1,) + img.shape[2:])

        else:
            (test_images, target) = testdata
            test_images /= div_val
            test_images_gpu = test_images.to(device)
            recon_data, y, _ = encodernet(test_images_gpu.to(device))
        
            if "4STACK" in args.model:
                test_images = test_images.reshape(20, 1, 84, 84)
                recon_data = recon_data.reshape(20, 1, 84, 84)

        vutils.save_image(torch.cat((torch.sigmoid(recon_data).cpu(), test_images.cpu()), 2),
                            "%s/%s/%s_%s.png" % (curr_dir, "Results", args.model, args.expname.upper() + "_" + str(args.kl_weight)))

        break
