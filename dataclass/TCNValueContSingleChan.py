from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
from IPython import embed
import torch

class TCNValueContSingleChan(BaseDataset):
    def __init__(self, root_dir, transform=None, value=True, episode=True, goal=False):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal)
        #self.value_thresh = value_thresh
        print(root_dir)
        #self.sample_next = sample_next

    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        
        
        if im_ind == self.limit_nps[file_ind][im_ind]:
            deltat = 0
        else:
            deltat = 1

        value = self.value_nps[file_ind][im_ind].astype(np.float32)

        #find the nearest value
        if im_ind != self.obs_nps[file_ind].shape[0]-1:
            getind = self.revind_nps[file_ind][im_ind + 1]
        else:
            getind = self.revind_nps[file_ind][im_ind - 1]
            
        targ = self.svalue_nps[file_ind][:, getind]
        assert(abs(targ[1] - im_ind) == 1)

        #print(value, targ[0], int(targ[1]))


        img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][im_ind + deltat].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][int(targ[1])].astype(np.float32), axis=0)]
        
        #return np.stack(img, axis=0), np.stack(value, axis=0), np.stack(episode, axis=0)
        return np.stack(img, axis=0)
