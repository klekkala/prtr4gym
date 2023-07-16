from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import bisect
from IPython import embed

class SingleChannelLSTM(BaseDataset):
    def __init__(self, root_dir, transform=None, max_seq_length = 1000):
        super().__init__(root_dir, transform, action=True, value=False, reward=True, episode=True, terminal=True, goal=False, use_lstm=True)
        self.max_seq_length = max_seq_length

    def __getitem__(self, item):
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = self.id_dict[file_ind][item]
        else:
            traj_ind = item - self.each_len[file_ind-1]
            im_ind = self.id_dict[file_ind][traj_ind]
        #print(file_ind, im_ind, self.max_len, self.each_len)
        #embed()
        last_img = self.limit_nps[file_ind][im_ind]
        
        #this is TxHxW
        traj = self.obs_nps[file_ind][im_ind:last_img].astype(np.float32)
        #this is TX2
        action = self.action_nps[file_ind][im_ind:last_img]

        trajimg = np.expand_dims(traj, 1)  # add channel dimension

        zs = np.zeros((self.max_seq_length - trajimg.shape[0],) + trajimg.shape[1:])
        trajimg = np.concatenate((trajimg, zs)) # padding

        target = trajimg[1:]  # offset
        target = np.concatenate((np.zeros((1,)+target.shape[1:]), target)) # padding
        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)
        action = np.concatenate((action, np.zeros((trajimg.shape[0]-action.shape[0],) + (action.shape[-1],)))) # padding

        return trajimg.astype(np.float32), target.astype(np.float32), action.astype(np.float32)
