from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import bisect
from IPython import embed

class CarlaFPVBEV(BaseDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform, action=False, reward=False, terminal=False, goal=False)


    def __getitem__(self, item):
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = item
        else:
            im_ind = item - self.each_len[file_ind-1]
        
        img = self.obs_nps[file_ind][im_ind]
        target = np.expand_dims(self.bev_nps[file_ind][im_ind][:, :, 0], axis=-1)
    
        img = np.moveaxis(img, -1, 0)
        target = np.moveaxis(target, -1, 0)

        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)

        return img, target