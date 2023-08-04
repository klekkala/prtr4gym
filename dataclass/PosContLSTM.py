from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
from IPython import embed
import torch

class PosContLSTM(BaseDataset):
    def __init__(self, root_dir, sample_next, max_seq_length=1000, transform=None):
        super().__init__(root_dir, transform, action=True, value=True, reward=True, episode=True, terminal=True, goal=False, use_lstm=True)
        #self.value_thresh = value_thresh
        print(root_dir)
        self.max_seq_length = max_seq_length
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

        assert(self.episode_nps[file_ind][im_ind] == self.episode_nps[file_ind][im_ind+deltat])

        curr_episode = self.episode_nps[file_ind][im_ind]
        
        start_ind = self.id_dict[file_ind][curr_episode]
        
        traj1img = np.expand_dims(self.obs_nps[file_ind][start_ind:im_ind-start_ind+1].astype(np.float32), axis=1)
        traj2img = np.expand_dims(self.obs_nps[file_ind][start_ind:im_ind-start_ind+deltat+1].astype(np.float32), axis=1)

        zs1 = np.zeros((self.max_seq_length - traj1img.shape[0],) + traj1img.shape[1:]).astype(np.float32)
        zs2 = np.zeros((self.max_seq_length - traj2img.shape[0],) + traj2img.shape[1:]).astype(np.float32)
        
        traj1img = np.concatenate((traj1img, zs1)) # padding
        traj2img = np.concatenate((traj2img, zs2)) # padding
    
        img = [traj1img, traj2img]
        return np.stack(img, axis=0)
