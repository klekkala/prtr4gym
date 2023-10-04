from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import os
import random
import bisect
from IPython import embed
import torch

class AtariVIPDataLoad(BaseDataset):
    def __init__(self, root_dir, max_len, transform=None, goal=True):
        super().__init__(root_dir, transform, action=True, value=False, reward=True, episode=True, terminal=True, goal=goal, use_lstm=False)
        #self.value_thresh = value_thresh
        self.min_len = 10
        self.max_len = max_len
        self.thresh_add = 5
        print(root_dir)
        print("max_len", max_len)


    def __getitem__(self, item):
        img, value, episode = [], [], []
        file_ind = bisect.bisect_right(self.each_len, item)
        if file_ind == 0:
            im_ind = item
        else:
            im_ind = item - self.each_len[file_ind-1]
        

        #start index of the episode
        #start_mark = self.id_dict[file_ind][im_ind]
        start_mark = im_ind

        #last index of the episode
        #actual last index: self.limit_nps[file_ind][start_ind]
        last_mark = self.limit_nps[file_ind][start_mark]

        #print(last_mark - start_mark)
        
        #if self.max_len == -1:
        #    self.max_len = last_mark

        #i don't like while loop in the dataloader
        while last_mark - start_mark < self.min_len:
            #before 13, 20
            im_ind -= np.random.randint(self.min_len, self.min_len + self.thresh_add)
            start_mark = im_ind
            last_mark = self.limit_nps[file_ind][start_mark]
        #random.randint(start_ind + 3, self.limit_nps[file_ind][start_ind])

        assert(last_mark - start_mark >= self.min_len)

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        # This seems to be a bug. start_ind is fixed. start_ind = np.random.randint(start_mark, last_mark-2) 
        start_ind = start_mark
        end_ind = np.random.randint(start_ind+4, min(start_ind + self.max_len, last_mark))

        mid_ind = np.random.randint(start_ind, end_ind)
        midplus = min(mid_ind+1, end_ind)

        #print(end_ind - start_ind)
        assert(self.min_len > 4)
        assert(end_ind - start_ind >= 3)
        assert(end_ind - start_ind <= self.max_len)


        #check the assertion later
        #assert(mid_int > start_ind and mid_int+1 < end_ind)
        #print(start_ind, mid_ind, end_ind)
        start_img = np.expand_dims(self.obs_nps[file_ind][start_ind].astype(np.float32), 0)
        last_img = np.expand_dims(self.obs_nps[file_ind][end_ind].astype(np.float32), 0)

        mid_img = np.expand_dims(self.obs_nps[file_ind][mid_ind].astype(np.float32), 0)
        midplus_img = np.expand_dims(self.obs_nps[file_ind][midplus].astype(np.float32), 0)


        
        return np.stack([start_img, mid_img, midplus_img, last_img], axis=0)
