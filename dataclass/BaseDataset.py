from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed


class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=None, action=False, reward=False, episode=False,
                 terminal=False, value=False, goal=False, truncated=True, use_lstm=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_lstm = use_lstm
        self.obs_nps = []
        self.revind_nps = []
        self.bev_nps = []
        self.each_len = []
        self.action_nps = []
        self.value_nps = []
        self.svalue_nps = []
        self.id_dict = []
        self.reward_nps = []
        self.episode_nps = []
        self.limit_nps = []
        self.terminal_nps = []
        self.value_map = []
        self.goal_nps = []
        self.aux_nps = []


        terminal_file = 'terminal_truncated.npy' if truncated else 'terminal'
        value_file = 'value_truncated.npy' if truncated else 'value'
        value_map = 'sorted_value_mapping_truncated' if truncated else 'value_mapping'
        id_dict_file = 'id_dict_truncated.npy' if truncated else 'id_dict.npy'
        limit_file = 'limit_truncated.npy' if truncated else 'limit.npy'
        episode_file = 'episode_truncated.npy' if truncated else 'episode.npy'
        
        print(self.root_dir)
        exten = ""
        if 'carla' in self.root_dir or 'trained' in self.root_dir or '3chan' in self.root_dir:
            exten = '.npy'

        for root, subdirs, files in os.walk(self.root_dir):
            print(files)
            if 'observation' + exten in files:
                #testing for airraid [:102585]
                self.obs_nps.append(np.load(root + '/observation' + exten, mmap_mode='r'))
                #self.obs_nps.append(np.load(root + '/observation' + exten, mmap_mode='r')[:10000,:,:,0])

                if action:
                    self.action_nps.append(np.load(root + '/action' + exten, mmap_mode='r'))

                if reward:
                    self.reward_nps.append(np.load(root + '/reward' + exten, mmap_mode='r'))

                if terminal:
                    self.terminal_nps.append(np.load(root + '/' + terminal_file, mmap_mode='r'))

                if value:
                    self.value_nps.append(np.load(root + '/' + value_file, mmap_mode='r'))
                    self.value_map.append(np.load(root + '/' + value_map, allow_pickle=True))
                    #self.value_map.append(np.load(root + '/' + exten, mmap_mode='r'))

                if episode:
                    self.episode_nps.append(np.load(root + '/' + episode_file, mmap_mode='r'))
                    self.limit_nps.append(np.load(root + '/' + limit_file, mmap_mode='r'))
                    print("episode files are: ", episode_file, limit_file)
                    ab = np.load(root + '/' + id_dict_file, allow_pickle=True)
                    self.id_dict.append(ab[()])

                if goal:
                    self.goal_nps.append(np.load(root + '/goal' + exten, mmap_mode='r'))
                    self.aux_nps.append(np.load(root + '/aux' + exten, mmap_mode='r'))




            elif 'fpv.npy' in files:
                print(root)
                print("lksjdflkjsalk;jflk;sajkal;sjdflk;jf")
                self.obs_nps.append(np.load(root + '/fpv.npy', mmap_mode='r'))
                self.bev_nps.append(np.load(root + '/bev.npy', mmap_mode='r'))

                if action:
                    self.action_nps.append(np.load(root + '/action.npy', mmap_mode='r'))

                if reward:
                    self.reward_nps.append(np.load(root + '/reward.npy', mmap_mode='r'))

                if terminal:
                    self.terminal_nps.append(np.load(root + '/terminal.npy', mmap_mode='r'))

                if episode:
                    self.episode_nps.append(np.load(root + '/episode.npy', mmap_mode='r'))
                    self.limit_nps.append(np.load(root + '/limit.npy', mmap_mode='r'))

                if self.use_lstm:
                    ab = np.load(root + '/id_dict.npy', allow_pickle=True)
                    self.id_dict.append(ab[()])

        
        if "rllib" in self.root_dir:
            for i in range(len(self.obs_nps)):
                self.obs_nps[i] = np.moveaxis(self.obs_nps[i], -1, 1)
        
        for i in range(len(self.obs_nps)):
            if self.use_lstm:
                if len(self.each_len) == 0:
                    self.each_len.append(self.episode_nps[i][-1])
                else:
                    self.each_len.append(self.episode_nps[i][-1] + self.each_len[-1])
            else:
                if len(self.each_len) == 0:
                    self.each_len.append(self.obs_nps[i].shape[0])
                else:
                    self.each_len.append(self.obs_nps[i].shape[0] + self.each_len[-1])

        self.max_len = self.each_len[-1]
        print("each_len", self.each_len)

        if max_len != None:
            print("setting max_len")
            self.lines = max_len
        
        self.lines = self.max_len
        self.num_files = len(self.obs_nps)
        print("truncated", truncated)
        
    def __len__(self):
        return self.lines - 1
