from torch.utils.data import Dataset, DataLoader
from dataclass.BaseDataset import BaseDataset
from collections import defaultdict
from PIL import Image
import numpy as np
import bisect
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
        self.maxlen = 10

    def lin_search(self, chose_ind, chose_game, thresh):
        minval = 100
        minind = -1

        if chose_ind == self.limit_nps[chose_game][chose_ind]:
            return chose_ind
        
        assert(chose_ind + 1 < min(chose_ind + self.max_len, self.limit_nps[chose_game][chose_ind]))
        for i in range(chose_ind, min(chose_ind + self.max_len, self.limit_nps[chose_game][chose_ind])):
            if minval > abs(thresh - self.value_nps[chose_game][i]):
                minind = i
                minval = abs(thresh - self.value_nps[chose_game][i])

        assert(minind > chose_ind)
        return minind

    def __getitem__(self, item):
        img, value, episode = [], [], []
        #need to change this to bisect
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        


        assert (len(self.obs_nps) > 1)
        #print(im_ind, deltat, self.limit_nps[file_ind][im_ind], self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]])
        assert(self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]] == 1)


        #self.action.append(self.action_nps[file_ind][im_ind].astype(np.uint8))
        value = self.value_nps[file_ind][im_ind].astype(np.float32)
        #next_value = rand_sample_until(self.treshold)
        
        delta = random.randint(im_ind+1, self.lin_search(im_ind, file_ind, value + self.threshold)) - im_ind
        assert(delta >= 0)

        next_value = self.value_nps[file_ind][im_ind + delta]

        assert(abs(next_value - value) <= self.threshold)

        #debug purposes
        #orig = self.svalue_nps[file_ind][:, int(self.revind_nps[file_ind][im_ind])]
        #print(self.value_nps[file_ind][im_ind], orig[0])
        #assert(self.value_nps[file_ind][im_ind] == orig[0])

        #randomly sample an index with the nearest value to the current value. (OLD method)
        #elind = int(self.revind_nps[file_ind][im_ind] + random.randint(-20, 20))

        #GOING TO OTHER GAME
        #randomly sample a game other than
        gamelist = list(range(len(self.obs_nps)))
        gamelist.remove(file_ind) 
        chose_game = random.choice(gamelist)

        #get the nearest value function in the sorted list
        val_list = list(self.value_map[chose_game].keys())

        val_ind = bisect.bisect_left(val_list, value)
        nearest_value = val_list[val_ind]
        assert(abs(nearest_value - value) < .05)

        #get the value in the corresponding game
        chose_ind = random.choice(self.value_map[chose_game][nearest_value])

        #get the delta
        assert(chose_ind <= self.limit_nps[chose_game][chose_ind])

        chose_delta = self.lin_search(chose_ind, chose_game, nearest_value + (next_value - value)) - chose_ind

        assert(chose_delta > 0)
        
        #if elind > self.max_len-1 or elind < 0 or abs(self.svalue_nps[file_ind][0, elind] - value) > self.threshold or elind == im_ind:
        #    if self.revind_nps[file_ind][im_ind] == 0:
        #        elind = int(self.revind_nps[file_ind][im_ind]) + 1
        #    else:
        #        elind = int(self.revind_nps[file_ind][im_ind]) - 1


        
        assert(abs(value - self.value_nps[file_ind][im_ind+delta]) <= self.threshold)
        assert(abs(nearest_value - self.value_nps[chose_game][chose_ind+chose_delta]) <= self.threshold)
        #print(value, targ[0], abs(elind - int(self.revind_nps[file_ind][im_ind])), self.threshold)



        #print(game_ind, file_ind)
        img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][im_ind + delta].astype(np.float32), axis=0)
                    ,np.expand_dims(self.obs_nps[chose_game][chose_ind + chose_delta].astype(np.float32), axis=0)
                    ,np.expand_dims(self.obs_nps[chose_game][chose_ind + chose_delta].astype(np.float32), axis=0)]
        
        return np.stack(img, axis=0)
