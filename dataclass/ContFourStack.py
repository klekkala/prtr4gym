from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
from IPython import embed
import torch

class ContFourStack(BaseDataset):
    def __init__(self, root_dir, transform=None, max_len=None):
        super().__init__(root_dir, transform, max_len=max_len, action=True, value=False, reward=True, episode=True, terminal=True, goal=False)
        #self.value_thresh = value_thresh
        print(root_dir)
        #self.sample_next = sample_next

    def __getitem__(self, item):
        file_ind = 0
        im_ind = item
        img = self.obs_nps[file_ind][im_ind].astype(np.float32)
        tar = self.action_nps[file_ind][im_ind].astype(np.uint8)
        #img = np.expand_dims(self.all_nps[file_ind][im_ind], axis=0).astype(np.float32) 
        #tar = img
        #img = torch.from_numpy(img)
        #tar = torch.from_numpy(tar)
        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)

        img = np.moveaxis(img, -1, 0)
        return img, tar

