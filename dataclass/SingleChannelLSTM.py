from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import bisect
from IPython import embed

class SingleChannelLSTM(BaseDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform, action=True, value=False, reward=True, episode=True, terminal=True, goal=False, use_lstm=True)

    def __getitem__(self, item):
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = self.id_dict[file_ind][item]
        #else:
        #    traj_ind = item - self.each_len[file_ind-1]
        #    im_ind = self.id_dict[file_ind][traj_ind]
        #print(file_ind, im_ind, self.max_len, self.each_len)
        #embed()
        #trajimg = self.obs_nps[file_ind][im_ind: self.limit_nps[file_ind][im_ind] - 1].astype(np.float32)
        trajimg = self.obs_nps[file_ind][im_ind: im_ind+5].astype(np.float32)

        target = trajimg

        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)

        return trajimg, target
