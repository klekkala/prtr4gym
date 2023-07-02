from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
from PIL import Image

class Atari101(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dict = defaultdict()
        # self.data = self.__loadfile(self.root_dir,self.root_dir)

        for name in os.listdir(self.root_dir):
            filename, ext = os.path.splitext(name)
            self.image_dict[filename] = name
        self.lines= len(self.image_dict)

    def __len__(self):
        return self.lines-1

    def __getitem__(self, item):

        img, target = Image.open(self.root_dir+"/"+self.image_dict[str(item)]),  Image.open(self.root_dir+"/"+self.image_dict[str(item)])

        if self.transform is not None:

            img = self.transform(img)
            target = self.transform(target)

        return img, target
