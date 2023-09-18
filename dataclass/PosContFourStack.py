from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
from IPython import embed
import torch

class PosContFourStack(BaseDataset):
    def __init__(self, root_dir, sample_next, transform=None, value=True, episode=True, goal=False):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal)
        #self.value_thresh = value_thresh
        print(root_dir)
        self.sample_next = sample_next

    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        
        assert (self.sample_next >= 0.0 and self.sample_next <= 1.0)
        
        if im_ind == self.limit_nps[file_ind][im_ind]:
            deltat = 0
        else:
            #deltat = 1
            deltat = np.random.geometric(1.0 - self.sample_next)

        #if it exceeds the limit.. normalize to the limit
        if im_ind + deltat > self.limit_nps[file_ind][im_ind]:
            deltat = int(np.random.uniform(1, self.limit_nps[file_ind][im_ind]-im_ind))

        
        #print(im_ind, deltat, self.limit_nps[file_ind][im_ind], self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]])
        assert(self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]] == 1)

        #print(im_ind, deltat, im_ind+deltat)
        #print(self.episode_nps[file_ind][im_ind], self.episode_nps[file_ind][im_ind+deltat], self.limit_nps[file_ind][im_ind])
        assert(self.episode_nps[file_ind][im_ind] == self.episode_nps[file_ind][im_ind+deltat])

        #self.action.append(self.action_nps[file_ind][im_ind].astype(np.uint8))
        #value = [self.value_nps[file_ind][im_ind].astype(np.float32)]
        #episode = [self.episode_nps[file_ind][im_ind].astype(np.int32)]


        img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][im_ind + deltat].astype(np.float32), axis=0)]
        
        #return np.stack(img, axis=0), np.stack(value, axis=0), np.stack(episode, axis=0)
        return np.stack(img, axis=0)
