from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed

class BaseDataset(Dataset):
    def __init__(self, root_dir, max_len, action=False, reward=False, terminal=False, goal=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.all_nps = []
        for root, subdirs, files in os.walk(self.root_dir):
            if 'observation' in files:
                print(root)
                self.all_nps.append(np.load(root + '/observation', mmap_mode='r'))
     
        self.lines = self.max_len
        self.num_files = len(self.all_nps)


    def __len__(self):
        return self.lines-1