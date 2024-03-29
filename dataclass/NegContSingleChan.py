from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed
import torch

class NegContSingleChan(BaseDataset):
    def __init__(self, root_dir, transform=None, goal=False, truncated=True):
        super().__init__(root_dir, transform, action=False, reward=False, terminal=False, goal=goal, truncated=truncated)


    def __getitem__(self, item):
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        img = self.obs_nps[file_ind][im_ind].astype(np.float32)
        tar = img
        
        return np.expand_dims(self.obs_nps[file_ind][im_ind], axis=0).astype(np.float32) 

        #tar = img
        #img = torch.from_numpy(img)
        #tar = torch.from_numpy(tar)
        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)
        #return img

