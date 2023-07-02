from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed

class SingleChannel(BaseDataset):
    def __init__(self, root_dir, max_len, action=False, reward=False, terminal=False, goal=False, transform=None):
        super().__init__(root_dir, max_len, action=False, reward=False, terminal=False, goal=False, transform=None)

    def __getitem__(self, item):
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        img = np.expand_dims(self.all_nps[file_ind][im_ind], axis=0).astype(np.float32)

        target = img

        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)

        return img, target

