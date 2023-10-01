from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed


class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=None, action=False, reward=False, episode=False,
                 terminal=False, value=False, goal=False, use_lstm=False):
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
        self.goal_nps = []

        exten = ""
        if 'carla' in self.root_dir or 'trained' in self.root_dir or '3chan' in self.root_dir:
            exten = '.npy'

        for root, subdirs, files in os.walk(self.root_dir):
            print(files)
            if 'observation' + exten in files:
                print(root)
                #testing for airraid [:102585]
                self.obs_nps.append(np.load(root + '/observation' + exten, mmap_mode='r'))
                # self.obs_nps.append(np.load(root + '/observation' + exten, mmap_mode='r')[:10000,:,:,0])
                if action:
                    self.action_nps.append(np.load(root + '/action' + exten, mmap_mode='r'))

                if reward:
                    self.reward_nps.append(np.load(root + '/reward' + exten, mmap_mode='r'))

                if terminal:
                    self.terminal_nps.append(np.load(root + '/terminal' + exten, mmap_mode='r'))

                if value:
                    exten = '.npy'
                    self.value_nps.append(np.load(root + '/value_truncated' + exten, mmap_mode='r'))
                    self.svalue_nps.append(np.load(root + '/sorted_value_truncated' + exten, mmap_mode='r'))
                    self.revind_nps.append(np.load(root + '/reversed_indices_truncated' + exten, mmap_mode='r'))
                    exten = ''

                if episode:
                    self.episode_nps.append(np.load(root + '/episode' + exten, mmap_mode='r'))
                    self.limit_nps.append(np.load(root + '/limit' + exten, mmap_mode='r'))

                if goal:
                    self.goal_nps.append(np.load(root + '/goal' + exten, mmap_mode='r'))

                if self.use_lstm:
                    ab = np.load(root + '/id_dict' + exten, allow_pickle=True)
                    self.id_dict.append(ab[()])

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
        print("blah blah")
        self.lines = self.max_len
        self.num_files = len(self.obs_nps)

        
    def __len__(self):
        return self.lines - 1
