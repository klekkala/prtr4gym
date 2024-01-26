from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
from IPython import embed
import torch

class TCNContSingleChan(BaseDataset):
    def __init__(self, root_dir, pos_distance, transform=None, value=False, episode=True, goal=False, truncated=True):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal, use_lstm=False, truncated=truncated)

        self.pos_distance = pos_distance
        assert (self.pos_distance <= 12)
        print("pos_distance", self.pos_distance)

        #self.sample_next = sample_next

    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)

        left_pos = self.pos_distance
        right_pos = self.pos_distance

        assert(self.limit_nps[file_ind][im_ind] - self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] >= 3)
        if self.pos_distance > 0:
            
            #set the right boundary
            if im_ind + right_pos > self.limit_nps[file_ind][im_ind] - 2:
                right_pos = max(self.limit_nps[file_ind][im_ind] - im_ind - 1, 0)

            #set the left boundary
            if im_ind - left_pos < self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] + 2:
                left_pos = max(im_ind - self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] - 1, 0)
            

            assert(im_ind + right_pos <= self.limit_nps[file_ind][im_ind])
            assert(im_ind - left_pos >= self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]])


            assert(right_pos >= 0)
            assert(left_pos >= 0)
            assert(right_pos != 0 or left_pos != 0)

            #randomly sample in the right direction
            if left_pos == 0:
                posarr = [None, random.randint(im_ind+1, im_ind+right_pos)]
                ind = 1

            #randomly sample in the left direction
            elif right_pos == 0:
                posarr = [random.randint(im_ind-left_pos, im_ind-1), None]
                ind = 0
            
            else:
                posarr = [random.randint(im_ind-left_pos, im_ind-1), random.randint(im_ind+1, im_ind+right_pos)]
                ind = random.randint(0, 1)

        #if pos_distance is -1 then sample from either side to the limit
        else:
            if abs(im_ind - self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]) < 2:
                posarr = [None, random.randint(im_ind+1, self.limit_nps[file_ind][im_ind]-1)]
                ind = 1

            elif abs(im_ind - self.limit_nps[file_ind][im_ind]) < 2:
                posarr = [random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-1), None]
                ind = 0

            else:
                posarr = [random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-1), random.randint(im_ind+1, self.limit_nps[file_ind][im_ind]-1)]
                ind = random.randint(0, 1)


        posind = posarr[ind]
        
        if ind == 0:
            assert(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] < posind)
            negind = random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]], posind-1)
        else:
            assert(posind < self.limit_nps[file_ind][im_ind])
            negind = random.randint(posind+1, self.limit_nps[file_ind][im_ind])


        #find d
        #+d, -d from the im_ind -> sample positive pair 
        #sample smth either from (start, -d) or (d, end)

        assert (abs(negind-im_ind) > abs(posind - im_ind))
        #sample anchor, positive and negative

        img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][posind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][negind].astype(np.float32), axis=0)]
        
        return np.stack(img, axis=0)
