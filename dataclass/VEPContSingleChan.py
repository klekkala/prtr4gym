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
    def __init__(self, root_dir, threshold, max_len, sample_batch, negtype, transform=None, value=True, episode=True, goal=False):
        super().__init__(root_dir, transform, action=True, value=value, reward=True, episode=episode, terminal=True, goal=goal, use_lstm=False)
        #self.value_thresh = value_thresh
        print(root_dir)
        self.max_len = max_len
        self.sample_batch = sample_batch

        self.threshold = threshold 
        self.negtype = negtype

        assert(self.threshold >= 0.05)

        #anywhere to the right
        assert(self.max_len < 10 and self.max_len >= -1.0 and self.max_len != 0.0)
        print(max_len)


    #search for an element in the other game that is closest
    #to the value difference in the first game
    def lin_search(self, chose_ind, chose_game, vlimit, loc_max_len):

        if loc_max_len > 0.0:
            end_val = min(chose_ind + loc_max_len, self.limit_nps[chose_game][chose_ind])
        else:
            end_val = self.limit_nps[chose_game][chose_ind]

        if chose_ind == end_val:
            return chose_ind
        elif chose_ind + 1 == end_val:
            return chose_ind+1
       
        #print(vlimit)
        if vlimit == -1:
            minind = random.randint(chose_ind+1, end_val)

        else:
            limind = chose_ind+1
            before = self.value_nps[chose_game][chose_ind]
            for i in range(chose_ind + 1, end_val):

                if self.value_nps[chose_game][i] > vlimit:
                    break
                else:
                    limind = i

            if limind-1 > chose_ind:
                minind = random.randint(chose_ind+1, limind-1)
            else:
                minind = chose_ind+1
            minval = self.value_nps[chose_game][minind]
       
            assert(minind <= limind)
            if minval > vlimit + 0.01:
                print("val exceeded", before, minval, vlimit)
        
        assert(minind > chose_ind)
        assert(minind <= end_val)
        return minind


    def find_neg(self, game, i, delta, start, end):
        choices = []
        #made a modification here
        #negative should be more far than the positive.
        #it need not be far as in 2*delta
        if i+delta + 1 <= end:
            if self.negtype == "nsame":
                return random.randint(i+delta+1, end)
            else:
                raise NotImplementedError
            choices.append(random.randint(i+delta+1, end))

        if i-delta-1 >= start:
            choices.append(random.randint(start, i-delta-1))
        
        if len(choices) == 0:
            print("bad")
            choices.append(random.randint(start, end))
        
        return random.choices(choices)[0]



    def __getitem__(self, item):
        img, value, episode = [], [], []
        allnegsamples, alldeltas, allinds = [], [], []

        #need to change this to bisect
        file_ind = int(item/1000000)
        allinds.append(item - (file_ind*1000000))
        
        #later
        #file_ind = bisect.bisect_right(self.each_len, item)
        #if file_ind == 0:
        #    im_ind = item
        #else:
        #    im_ind = item - self.each_len[file_ind-1]       
        #allinds.append(im_ind)

        assert (len(self.obs_nps) > 1)
        #print(im_ind, deltat, self.limit_nps[file_ind][im_ind], self.terminal_nps[file_ind][self.limit_nps[file_ind][im_ind]])
        assert(self.terminal_nps[file_ind][self.limit_nps[file_ind][allinds[-1]]] == 1)


        #self.action.append(self.action_nps[file_ind][im_ind].astype(np.uint8))
        value = self.value_nps[file_ind][allinds[-1]].astype(np.float32)
        
        if value == 0.0:
            #replace the element if the value is 0.0
            while value == 0:
                curind = random.randint(0, 1000000)
                value = self.value_nps[file_ind][curind].astype(np.float32)
        
            allinds[-1] = curind

        assert(value != 0.0)
        episode1_start = self.id_dict[file_ind][self.episode_nps[file_ind][allinds[-1]]]

        if self.threshold == -1:
            delta = self.lin_search(allinds[-1], file_ind, -1, int(self.max_len)) - allinds[-1]
        else:
            #FIND THE ELEMENT THAT HAS THE CLOSEST VALUE TO VALUE+THRESHOLD WITHIN THE MAX DISTANCE MEASURE
            delta = self.lin_search(allinds[-1], file_ind, value + self.threshold, int(self.max_len)) - allinds[-1]
        

        alldeltas.append(delta)
        allnegsamples.append(self.find_neg(file_ind, allinds[-1], delta, episode1_start, self.limit_nps[file_ind][allinds[-1]]))
        assert(alldeltas[-1] >= 0)
        assert(allnegsamples[-1] >= episode1_start and allnegsamples[-1] <= self.limit_nps[file_ind][allinds[-1]])
        assert(self.limit_nps[file_ind][allinds[-1]] == self.limit_nps[file_ind][allinds[-1]+alldeltas[-1]])
        assert(self.limit_nps[file_ind][episode1_start] == self.limit_nps[file_ind][allinds[-1]] == self.limit_nps[file_ind][allinds[-1]+alldeltas[-1]])
        next_value = self.value_nps[file_ind][allinds[-1] + alldeltas[-1]]




        #the next 1 and 3 lines were enforced until 1/27/2024 18:18PM. After this time, the condition was removed
        #randomly sample n-1 games other than the current game
        gamelist = list(range(len(self.obs_nps)))
        #gamelist.remove(file_ind) 
        chose_games = random.choices(gamelist, k=self.sample_batch-1)
        chose_games = [file_ind] + chose_games
        
        
        
        for i in range(1, len(chose_games)):

            #get the nearest value function in the sorted list
            val_list = list(self.value_map[chose_games[i]].keys())

            val_ind = bisect.bisect_left(val_list, value)
            nearest_value = val_list[val_ind]
            #print(nearest_value, value)
        
            if nearest_value - value > .1:
                print(value, nearest_value, chose_games[0], chose_games[i])

            #get the value in the corresponding game
            allinds.append(random.choice(self.value_map[chose_games[i]][nearest_value]))
            assert(abs(nearest_value - self.value_nps[chose_games[i]][allinds[-1]]) < .05)

            #get the delta
            assert(allinds[-1] <= self.limit_nps[chose_games[i]][allinds[-1]])

            episode_start = self.id_dict[chose_games[i]][self.episode_nps[chose_games[i]][allinds[-1]]]


            if self.threshold == -1:
                alldeltas.append(self.lin_search(allinds[-1], chose_games[i], -1, int(self.max_len)) - allinds[-1])
            else:
                alldeltas.append(self.lin_search(allinds[-1], chose_games[i], nearest_value + self.threshold, int(self.max_len)) - allinds[-1])
            

            assert(alldeltas[-1] >= 0)
            assert(self.limit_nps[chose_games[i]][allinds[-1]] == self.limit_nps[chose_games[i]][allinds[-1]+alldeltas[-1]])

            
            allnegsamples.append(self.find_neg(chose_games[i], allinds[-1], alldeltas[-1], episode_start, self.limit_nps[chose_games[i]][allinds[-1]]))

            
            assert(self.limit_nps[chose_games[i]][episode_start] == self.limit_nps[chose_games[i]][allinds[-1]] == self.limit_nps[chose_games[i]][allinds[-1]+alldeltas[-1]])
            assert(allnegsamples[-1] >= episode_start and allnegsamples[-1] <= self.limit_nps[chose_games[i]][allinds[-1]])





        img = [0]*self.sample_batch*3
        assert(len(chose_games) == self.sample_batch)
        #print(allinds, alldeltas, allnegsamples)
        for i in range(self.sample_batch):
            #print(game_ind, file_ind)       
            #anchor
            img[i] = np.expand_dims(self.obs_nps[chose_games[i]][allinds[i]].astype(np.float32), axis=0)
            #positive
            img[i+self.sample_batch] = np.expand_dims(self.obs_nps[chose_games[i]][allinds[i] + alldeltas[i]].astype(np.float32), axis=0)
            #negative
            img[i+2*self.sample_batch] = np.expand_dims(self.obs_nps[chose_games[i]][allnegsamples[i]].astype(np.float32), axis=0)
 

        return np.stack(img, axis=0)
