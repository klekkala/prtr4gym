from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import bisect
from IPython import embed
import torchvision.utils as vutils
import torch

class SingleChannelLSTM(BaseDataset):
    def __init__(self, root_dir, transform=None, max_seq_length=1000):
        super().__init__(root_dir, transform, action=True, value=False, reward=True, episode=True, terminal=True,
                         goal=False, use_lstm=True)
        self.max_seq_length = max_seq_length

    def __getitem__(self, item):
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = self.id_dict[file_ind][item]
        else:
            traj_ind = item - self.each_len[file_ind - 1]
            im_ind = self.id_dict[file_ind][traj_ind]
        # print(file_ind, im_ind, self.max_len, self.each_len)
        # embed()
        last_img = self.limit_nps[file_ind][im_ind]
        # this is TxHxW
        length = int(last_img - im_ind) #int((last_img - im_ind) / 2) + 1
        traj = self.bev_nps[file_ind][im_ind:last_img+1, :, :, 0].astype(np.float32)
        # this is TX2
        action = self.action_nps[file_ind][im_ind:last_img]
        trajimg = np.expand_dims(traj, 1)  # add channel dimension

        #pads = np.tile(source[-1], (self.max_seq_length - source.shape[0], 1, 1, 1))
        #source = np.concatenate((source, pads))  # padding

        target = trajimg if length < 1 else trajimg[1:]  # offset
        action = np.zeros((1, 2)) if length < 1 else action
        source = trajimg if length < 1 else trajimg[:-1]#[:length]
        #target = np.concatenate((target, np.tile(target[-1], (self.max_seq_length-target.shape[0], 1, 1, 1))))  # padding
        # if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)
        #action = np.concatenate(
        #    (action, np.zeros((self.max_seq_length - action.shape[0],) + (action.shape[-1],))))  # padding

        return source.astype(np.float32), target.astype(np.float32), action.astype(np.float32)

