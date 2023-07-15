from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed

class ContBaseDataset(Dataset):
    def __init__(self, root_dir, num_games, transform=None, action=False, reward=False, terminal=False, goal=False):
        self.root_dir = root_dir
        self.transform = transform
        self.num_games = num_games
        self.obs_nps = [[] for _ in self.num_games]
        self.action_nps = [[] for _ in self.num_games]
        self.reward_nps = [[] for _ in self.num_games]
        self.terminal_nps = [[] for _ in self.num_games]
        self.goal_nps = [[] for _ in self.num_games]
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

                if goal:
                    self.goal_nps.append(np.load(root + '/goal', mmap_mode='r'))


        self.lines = self.max_len
        self.num_files = len(self.obs_nps)


    def __len__(self):
        return self.lines-1