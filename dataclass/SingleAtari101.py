from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
from PIL import Image

class SingleAtari101(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dict = defaultdict()
        # self.data = self.__loadfile(self.root_dir,self.root_dir)

        self.allfiles = os.listdir(self.root_dir)

        self.lines= len(os.listdir(root_dir))

    def __len__(self):
        return self.lines-1

    def __getitem__(self, item):

        img = Image.open(self.root_dir + "/" + self.allfiles[item])
        target = img

        if self.transform is not None:

            img = self.transform(img)
            target = self.transform(target)

        return img, target
