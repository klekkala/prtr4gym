from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed

class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, action=False, reward=False, episode=False, terminal=False, value=False, goal=False, use_lstm=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_lstm = use_lstm
        self.obs_nps = []
        self.each_len = []
        self.action_nps = []
        self.value_nps = []
        self.id_dict = []
        self.reward_nps = []
        self.episode_nps = []
        self.limit_nps = []
        self.terminal_nps = []
        self.goal_nps = []
        exten = ""
        if 'carla' in self.root_dir:
            exten = '.npy'
        for root, subdirs, files in os.walk(self.root_dir):

            if 'observation' + exten in files:
                print(root)
                self.obs_nps.append(np.load(root + '/observation' + exten, mmap_mode='r'))

                if action:
                    self.action_nps.append(np.load(root + '/action' + exten, mmap_mode='r'))

                if reward:
                    self.reward_nps.append(np.load(root + '/reward' + exten, mmap_mode='r'))

                if terminal:
                    self.terminal_nps.append(np.load(root + '/terminal' + exten, mmap_mode='r'))

                if value:
                    self.value_nps.append(np.load(root + '/value' + exten, mmap_mode='r'))
                
                if episode:
                    self.episode_nps.append(np.load(root + '/episode' + exten, mmap_mode='r'))

                if goal:
                    self.goal_nps.append(np.load(root + '/goal' + exten, mmap_mode='r'))

                if self.use_lstm:
                    ab = np.load(root + '/id_dict' + exten, allow_pickle=True)
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


        #self.max_len = self.each_len[-1]
        self.max_len = 100
        self.lines = self.max_len
        self.num_files = len(self.obs_nps)


    def __len__(self):
        return self.lines-1