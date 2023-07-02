from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed
import torch

class FourStackAtari(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = 1000000
        self.all_nps = []

        for np_file in os.listdir(self.root_dir):
            print(np_file)
            #expert
            self.all_nps.append(np.load(self.root_dir + '/' + np_file, mmap_mode='r'))
            #expert
            self.all_act.append(np.load(self.root_dir + '/' + np_file, mmap_mode='r'))
        
        self.lines = self.max_len
        self.num_files = len(self.all_nps)


    def __len__(self):
        return self.lines-1

    def __getitem__(self, item):
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        img = self.all_nps[file_ind][im_ind].astype(np.float32)
        tar = self.all_nps[file_ind][im_ind].astype(np.float32)
        img = torch.from_numpy(img)
        tar = torch.from_numpy(tar)
        #if self.transform is not None:
        #    img = self.transform(img)
        #    target = self.transform(target)

        return img, tar

