from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed
import torch

class ContFourStack(BaseDataset):
    def __init__(self, root_dir, max_len, transform=None):
        super().__init__(root_dir, max_len, transform, action=True, reward=True, terminal=True, goal=False)
        #self.value_thresh = value_thresh
        print(root_dir)


    def __getitem__(self, item):
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        img = self.obs_nps[file_ind][im_ind].astype(np.float32)
        action = self.action_nps[file_ind][im_ind].astype(np.uint8)

        img = np.moveaxis(img, -1, 0)

        #get img2 in a similar fashion
        #img2
        
        ################# Process the data here to decide which elements in the 2 batches are positive pairs ########

        #compute the values (sum of discounted rewards) of both the states
        #find the terminal flags of both the episodes and compute the reward until then
        #term1 = self.all_nps[file_ind][im_ind]
        #compute the values.

        #vf = np.arange(1000000)
        #gamma=0.95
        #for i in range(1000000):
        #    sumval = 0
        #    output = np.sum(arr[i:] * a ** powers[: arr.size - i]) for i in range(arr.size)
       
        #    vf[i] = sumval
        #powers = np.arange(arr.size)
        

        #v1 = 
        #v2 = 



        #decide if 2 of them form a pair
        #assert(a1 < 6 and a2 < 6)
        #if a1 != a2 or abs(v1 - v2) < self.value_thresh:
        #    return img1, img2, 0
        #
        #else:
        #    return img1, img2, 1
        #return 2 batches and if they are a positive/negative pair.
        return img, action

