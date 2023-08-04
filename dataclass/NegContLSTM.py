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
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        curr_episode = self.episode_nps[file_ind][im_ind]

        start_ind = self.id_dict[file_ind][curr_episode]
        
        trajimg = self.obs_nps[file_ind][start_ind:im_ind-start_ind+1].astype(np.float32)
        zs = np.zeros((self.max_seq_length - trajimg.shape[0],) + trajimg.shape[1:])
        trajimg = np.concatenate((trajimg, zs)) # padding

        return trajimg

