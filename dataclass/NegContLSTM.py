from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed
import torch

class NegContLSTM(BaseDataset):
    def __init__(self, root_dir, max_seq_length=1000, transform=None):
        super().__init__(root_dir, transform, action=False, reward=False, terminal=False, episode=True, goal=False, use_lstm=True)
        self.max_seq_length = max_seq_length

    def __getitem__(self, item):
        file_ind = 0
        im_ind = item
        #file_ind = int(item/1000000)
        #im_ind = item - (file_ind*1000000)
        
        
        #start index of the episode
        start_ind = self.id_dict[file_ind][im_ind]
        
        #last index of the episode
        last_ind = self.limit_nps[file_ind][start_ind]

        inputtraj = np.expand_dims(self.obs_nps[file_ind][start_ind:last_ind+1].astype(np.float32), axis=1)
        try:
            zs = np.zeros((self.max_seq_length - inputtraj.shape[0],) + inputtraj.shape[1:]).astype(np.float32)
        except:
            print(inputtraj.shape)
        inputtraj = np.concatenate((inputtraj, zs)) # padding

        return inputtraj

