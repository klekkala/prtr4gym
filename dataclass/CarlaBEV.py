from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import bisect
from IPython import embed

class CarlaBEV(BaseDataset):
    def __init__(self, root_dir, max_len=None, transform=None):
        super().__init__(root_dir, max_len, transform, action=False, reward=False, terminal=False, goal=False)


    def __getitem__(self, item):
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = item
        else:
            im_ind = item - self.each_len[file_ind-1]
        #print(file_ind, im_ind, self.max_len, self.each_len)
        #embed()
        img = np.expand_dims(self.obs_nps[file_ind][im_ind], axis=0).astype(np.float32)

        target = img

        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)

        return img, target

