from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
import bisect
from IPython import embed
import torch

class VIPDataLoad(BaseDataset):
    def __init__(self, root_dir, transform=None, goal=True):
        super().__init__(root_dir, transform, action=True, value=False, reward=True, episode=True, terminal=True, goal=goal, use_lstm=True)
        #self.value_thresh = value_thresh
        print(root_dir)


    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = item
        else:
            im_ind = item - self.each_len[file_ind-1]
        

        #start index of the episode
        start_mark = self.id_dict[file_ind][im_ind]

        #last index of the episode
        #actual last index: self.limit_nps[file_ind][start_ind]
        last_mark = self.limit_nps[file_ind][start_mark]
        
        #random.randint(start_ind + 3, self.limit_nps[file_ind][start_ind])

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(start_mark, last_mark-2)  
        end_ind = np.random.randint(start_ind+1, last_mark)

        mid_int = np.random.randint(start_ind, end_ind)
        midplus = min(mid_int+1, end_ind)


        #check the assertion later
        #assert(mid_int > start_ind and mid_int+1 < end_ind)
        
        start_img = np.moveaxis(self.obs_nps[file_ind][start_ind].astype(np.float32), -1, 0)
        last_img = np.moveaxis(self.obs_nps[file_ind][end_ind].astype(np.float32), -1, 0)

        mid_img = np.moveaxis(self.obs_nps[file_ind][mid_int].astype(np.float32), -1, 0)
        midplus_img = np.moveaxis(self.obs_nps[file_ind][midplus].astype(np.float32), -1, 0)


        
        return np.stack([start_img, mid_img, midplus_img, last_img], axis=0)
