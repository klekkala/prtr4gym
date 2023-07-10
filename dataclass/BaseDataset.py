from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed

class BaseDataset(Dataset):
    def __init__(self, root_dir, max_len, transform=None, action=False, reward=False, episode=False, terminal=False, value=False, goal=False):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.obs_nps = []
        self.action_nps = []
        self.value_nps = []
        self.reward_nps = []
        self.episode_nps = []
        self.terminal_nps = []
        self.goal_nps = []
        for root, subdirs, files in os.walk(self.root_dir):

            if 'observation' in files:
                print(root)
                self.obs_nps.append(np.load(root + '/observation', mmap_mode='r'))
     
                if action:
                    self.action_nps.append(np.load(root + '/action', mmap_mode='r'))

                if reward:
                    self.reward_nps.append(np.load(root + '/reward', mmap_mode='r'))

                if terminal:
                    self.terminal_nps.append(np.load(root + '/terminal', mmap_mode='r'))

                if value:
                    self.value_nps.append(np.load(root + '/value', mmap_mode='r'))
                
                if episode:
                    self.episode_nps.append(np.load(root + '/episode', mmap_mode='r'))

                if goal:
                    self.goal_nps.append(np.load(root + '/goal', mmap_mode='r'))


        self.lines = self.max_len
        self.num_files = len(self.obs_nps)


    def __len__(self):
        return self.lines-1