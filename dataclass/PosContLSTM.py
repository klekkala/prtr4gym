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
        file_ind = 0
        im_ind = item
        #file_ind = int(item/1000000)
        #im_ind = item - (file_ind*1000000)
       
        #start index of the episode
        start_ind = self.id_dict[file_ind][item]

        #last index of the episode
        last_ind = self.limit_nps[file_ind][start_ind]
        

        
        inputtraj = np.expand_dims(self.obs_nps[file_ind][start_ind:last_ind+1].astype(np.float32), axis=1)
        zs = np.zeros((self.max_seq_length - inputtraj.shape[0],) + inputtraj.shape[1:]).astype(np.float32)
        
        targettraj = np.concatenate((inputtraj[1:,:, :,:], inputtraj[-1:, :, :, :]), axis=0)
        inputtraj = np.concatenate((inputtraj, zs)) # padding
        targettraj = np.concatenate((targettraj, zs)) # padding
    
        return np.stack([inputtraj, targettraj], axis=0)
