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
    def __init__(self, root_dir, pos_distance, transform=None, value=False, episode=True, goal=False):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal, use_lstm=False)
        self.pos_distance = pos_distance
        assert (self.pos_distance < 5)
        print("pos_distance", self.pos_distance)
        #self.sample_next = sample_next

    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        
        if self.pos_distance > 0:
            if im_ind + self.pos_distance >= self.limit_nps[file_ind][im_ind] and im_ind - self.pos_distance <= self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]:
                print("bad")
                img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0)]
                return np.stack(img, axis=0)
            
            #if the current index is close to the end, then only sample from the left
            if im_ind + self.pos_distance > self.limit_nps[file_ind][im_ind] - 2:
                if im_ind-self.pos_distance <= self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]:
                    print("bad")
                    posarr = [im_ind, None]
                else:
                    posarr = [random.randint(max(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-self.pos_distance), im_ind-1), None]
                ind = 0

            #if the current index is close to the start, then only sample from the right
            elif im_ind - self.pos_distance < self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] + 2:
                if im_ind+self.pos_distance >= self.limit_nps[file_ind][im_ind]:
                    print("bad")
                    posarr = [None, im_ind]
                else:
                    posarr = [None, random.randint(im_ind+1, min(self.limit_nps[file_ind][im_ind]-1, im_ind+self.pos_distance))]
                ind = 1
            
            #if the current index is in the middle, then sample posind from either of the direction
            #NOTE THAT WE ARE ASSUMING POS AND NEG TO SAMPLE IN THE SAME DIRECTION
            else:
                #marker is im_ind. check if im_ind > 0 and im_ind < max_len
                posarr = [random.randint(max(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-self.pos_distance), im_ind-1), random.randint(im_ind+1, min(self.limit_nps[file_ind][im_ind]-1, im_ind+self.pos_distance))]
                ind = random.randint(0, 1)

        #if pos_distance is -1 then sample from either side to the limit
        else:
            if abs(im_ind-self.limit_nps[file_ind][im_ind]) < abs(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]] - im_ind):
                posarr = [random.randint(self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]+1, im_ind-1), None]
                ind = 0
            else:
                posarr = [None, random.randint(im_ind+1, self.limit_nps[file_ind][im_ind]-1)]
                ind = 1


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
