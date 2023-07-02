from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
import numpy as np
import os
from IPython import embed
import torch

class ThreeChannel(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = 3000000
        self.all_nps = []

        for root, subdirs, files in os.walk(self.root_dir):
            if len(subdirs) == 3:
                for eachdir in subdirs:
                    print(eachdir)
                    self.all_nps.append(np.load(root + '/' + eachdir + '/observation', mmap_mode='r'))
        
        self.lines = self.max_len
        self.num_files = len(self.all_nps)


    def __len__(self):
        return self.lines-1

    def __getitem__(self, item):
        file_ind = int(item/1000000)
        im_ind = item - (file_ind*1000000)
        img = self.all_nps[file_ind][im_ind].astype(np.float32)
        preimg = np.repeat(img[np.newaxis, :, :], 3, axis=0)

        #target = img

        #if self.transform is not None:
        #    print(self.transform)
        #    img = self.transform(preimg)
        #    target = self.transform(preimg)
        #    print(img.shape, target.shape, preimg.shape)

        img = torch.from_numpy(preimg)
        #target = torch.from_numpy(preimg)
        return img, img

