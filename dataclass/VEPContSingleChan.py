from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
from IPython import embed
import torch

class VEPContSingleChan(BaseDataset):
    def __init__(self, root_dir, threshold, transform=None, value=True, episode=True, goal=False):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal)
        #self.value_thresh = value_thresh
        print(root_dir)
        self.threshold = threshold

    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        



        """
        #print(im_ind, deltat, self.limit_nps[file_ind][im_ind], self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]])
        assert(self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]] == 1)

        #print(im_ind, deltat, im_ind+deltat)
        #print(self.episode_nps[file_ind][im_ind], self.episode_nps[file_ind][im_ind+deltat], self.limit_nps[file_ind][im_ind])
        assert(self.episode_nps[file_ind][im_ind] == self.episode_nps[file_ind][im_ind+deltat])
        """
        #self.action.append(self.action_nps[file_ind][im_ind].astype(np.uint8))
        value = self.value_nps[file_ind][im_ind].astype(np.float32)
        #episode = [self.episode_nps[file_ind][im_ind].astype(np.int32)]

        #randomly sample an index with the nearest value to the current value.
        elind = int(self.revind_nps[file_ind][im_ind] + random.randint(-20, 20))

        #debug purposes
        orig = self.svalue_nps[file_ind][:, int(self.revind_nps[file_ind][im_ind])]
        #print(self.value_nps[file_ind][im_ind], orig[0])
        assert(self.value_nps[file_ind][im_ind] == orig[0])
        
        if elind > self.max_len-1 or elind < 0 or abs(self.svalue_nps[file_ind][0, elind] - value) > self.threshold or elind == im_ind:
            if self.revind_nps[file_ind][im_ind] == 0:
                elind = int(self.revind_nps[file_ind][im_ind]) + 1
            else:
                elind = int(self.revind_nps[file_ind][im_ind]) - 1

        #get the data from the index.
        targ = self.svalue_nps[file_ind][:, elind]
        
        #print(abs(targ[0] - value), elind, int(self.revind_nps[file_ind][im_ind]))
        assert(abs(targ[0] - value) <= self.threshold)
        #print(value, targ[0], abs(elind - int(self.revind_nps[file_ind][im_ind])), self.threshold)

        if targ.shape[0] == 3:
            game_ind = int(targ[1])
            t_ind = int(targ[2])
        elif targ.shape[0] == 2:
            game_ind = 0
            t_ind = int(targ[1])
        else:
            raise ValueError

        #print(game_ind, file_ind)
        img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[game_ind][t_ind].astype(np.float32), axis=0)]
        
        #return np.stack(img, axis=0), np.stack(value, axis=0), np.stack(episode, axis=0)
        return np.stack(img, axis=0)
