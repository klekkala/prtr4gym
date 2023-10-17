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
    def __init__(self, root_dir, threshold, max_len, transform=None, value=True, episode=True, goal=False):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal, use_lstm=True)
        #self.value_thresh = value_thresh
        print(root_dir)
        self.threshold = threshold
        self.max_len = max_len
        if self.max_len == -1:
            print(print("max_len -1"))
        elif self.max_len < 1.0:
            assert(self.max_len > 0.0 and self.max_len <= .3)
        else:
            ###fix this
            assert(self.max_len < 20)
        print(max_len)
        #add an assertion to maxlen

    def lin_search(self, chose_ind, chose_game, vlimit, loc_max_len):

        if chose_ind == min(chose_ind + loc_max_len, self.limit_nps[chose_game][chose_ind]):
            return chose_ind
        elif chose_ind + 1 == min(chose_ind + loc_max_len, self.limit_nps[chose_game][chose_ind]):
            return chose_ind+1
        
        assert(chose_ind + 1 < min(chose_ind + loc_max_len, self.limit_nps[chose_game][chose_ind]))
        minval = self.value_nps[chose_game][chose_ind+1]
        minind = chose_ind+1
        for i in range(chose_ind + 1, min(chose_ind + loc_max_len, self.limit_nps[chose_game][chose_ind])):

            if minval > abs(vlimit - self.value_nps[chose_game][i]):
                minind = i
                minval = abs(vlimit - self.value_nps[chose_game][i])

        assert(minind > chose_ind)
        assert(minind <= chose_ind + loc_max_len)
        return minind


    def find_neg(self, game, i, delta, start, end):
        choices = []
        #made a modification here
        #negative should be more far than the positive.
        #it need not be far as in 2*delta
        if i+delta + 1 <= end:
            choices.append(random.randint(i+delta+1, end))
        if i-delta-1 >= start:
            choices.append(random.randint(start, i-delta-1))
        
        #instead of this
        #assert(len(choices) > 0)
        #use this
        if len(choices) == 0:
            print("bad")
            choices.append(random.randint(start, end))
        
        return random.choices(choices)[0]



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
        

        ##I CHANGED THIS!!
        if self.max_len != -1:
            episode1_start = self.id_dict[file_ind][self.episode_nps[file_ind][im_ind]]
            game1_max_len = self.limit_nps[file_ind][im_ind] - episode1_start + 1
            game1_distance_thresh = self.max_len*game1_max_len if self.max_len < 1.0 else self.max_len
        else:
            game1_distance_thresh = -1
        delta = random.randint(im_ind, self.lin_search(im_ind, file_ind, value + self.threshold, int(game1_distance_thresh))) - im_ind
        assert(delta >= 0)
        assert(delta <= self.max_len*game1_max_len+1)

        next_value = self.value_nps[file_ind][im_ind + delta]

        #assert(next_value > value)
        #assert(abs(next_value - value) <= self.threshold)

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
        assert(abs(nearest_value - self.value_nps[chose_game][chose_ind]) < .05)

        #get the delta
        assert(chose_ind <= self.limit_nps[chose_game][chose_ind])

        episode2_start = self.id_dict[chose_game][self.episode_nps[chose_game][chose_ind]]
        game2_max_len = self.limit_nps[chose_game][chose_ind] - episode2_start + 1
        game2_distance_thresh = self.max_len*game2_max_len if self.max_len < 1.0 else self.max_len
        chose_delta = self.lin_search(chose_ind, chose_game, nearest_value + abs(next_value - value), int(game2_distance_thresh)) - chose_ind

        assert(chose_delta <= self.max_len*game2_max_len+1)

        assert(chose_delta >= 0)
        
        #if elind > self.max_len-1 or elind < 0 or abs(self.svalue_nps[file_ind][0, elind] - value) > self.threshold or elind == im_ind:
        #    if self.revind_nps[file_ind][im_ind] == 0:
        #        elind = int(self.revind_nps[file_ind][im_ind]) + 1
        #    else: 
        #        elind = int(self.revind_nps[file_ind][im_ind]) - 1


        assert(self.limit_nps[file_ind][im_ind] == self.limit_nps[file_ind][im_ind+delta])
        assert(self.limit_nps[chose_game][chose_ind] == self.limit_nps[chose_game][chose_ind+chose_delta])
        #print(value, next_value, delta, self.max_len*game1_max_len, nearest_value, self.value_nps[chose_game][chose_ind + chose_delta], chose_delta, self.max_len*game2_max_len)
        #print(value, next_value, nearest_value, self.value_nps[chose_game][chose_ind + chose_delta])
        #print(delta, chose_delta)
        
        negsample1 = self.find_neg(file_ind, im_ind, delta, episode1_start, self.limit_nps[file_ind][im_ind])
        negsample2 = self.find_neg(chose_game, chose_ind, chose_delta, episode2_start, self.limit_nps[chose_game][chose_ind])
        #print(negsample1, im_ind, im_ind+delta)

        #check if deltas are within permissible range
        #assert(delta <= .3*game1_max_len)
        #assert(chose_delta <= .3*game2_max_len)

        assert(self.limit_nps[file_ind][episode1_start] == self.limit_nps[file_ind][im_ind] == self.limit_nps[file_ind][im_ind+delta])
        assert(self.limit_nps[chose_game][episode2_start] == self.limit_nps[chose_game][chose_ind] == self.limit_nps[chose_game][chose_ind+chose_delta])

        assert(negsample1 < im_ind-delta or negsample1 > im_ind+delta)
        assert(negsample2 < chose_ind-chose_delta or negsample2 > chose_ind+chose_delta)
        assert(negsample1 >= episode1_start and negsample1 <= self.limit_nps[file_ind][im_ind])
        assert(negsample2 >= episode2_start and negsample2 <= self.limit_nps[chose_game][chose_ind])

        #print(game_ind, file_ind)
        img = [np.expand_dims(self.obs_nps[file_ind][im_ind].astype(np.float32), axis=0), np.expand_dims(self.obs_nps[file_ind][im_ind + delta].astype(np.float32), axis=0)
                    ,np.expand_dims(self.obs_nps[chose_game][chose_ind + chose_delta].astype(np.float32), axis=0)
                    ,np.expand_dims(self.obs_nps[chose_game][chose_ind + chose_delta].astype(np.float32), axis=0)
                    ,np.expand_dims(self.obs_nps[file_ind][negsample1].astype(np.float32), axis=0)
                    ,np.expand_dims(self.obs_nps[chose_game][negsample2].astype(np.float32), axis=0)]
        
        return np.stack(img, axis=0)
