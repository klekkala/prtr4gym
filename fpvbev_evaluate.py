import numpy as np
import torch.nn as nn
import os
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")

class BEVEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(BEVEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Linear(h_dim, z)

    def forward(self, x):
        return self.fc(self.encoder(x))

class ResNet(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embed_size)


    def forward(self, image):
        out = self.resnet(image)
        return out

class Encoder(nn.Module):
    def __init__(self, encoder_path):
        super().__init__()
        self.fpvencoder = ResNet(512).to(device)
        self.bevencoder = BEVEncoder(channel_in=1, ch=32, h_dim=1024, z=512).to(device)

        # load models
        checkpoint = torch.load(encoder_path, map_location="cpu")
        print("epoch:", checkpoint['epoch'])
        self.fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
        self.bevencoder.load_state_dict(checkpoint['bev_state_dict'])
        self.fpvencoder.eval()
        for param in self.fpvencoder.parameters():
            param.requires_grad = False
        self.bevencoder.eval()
        for param in self.bevencoder.parameters():
            param.requires_grad = False

        # read anchor images and convert to latent representations
        self.anchors_lr = []
        self.anchors = []
        for i in range(1, 101):
            im = cv2.imread(os.path.join("/lab/kiran/img2cmd/test", str(i)+'00_.jpg'), cv2.IMREAD_GRAYSCALE)
            self.anchors.append(im)
            with torch.no_grad():
                im = np.expand_dims(im, axis=(0, 1))
                im = torch.tensor(im).to(device) / 255.0
                self.anchors_lr.append(self.bevencoder(im)[0].cpu().numpy())
        self.anchors_lr = np.array(self.anchors_lr)
        self.anchors_lr = torch.tensor(self.anchors_lr).to(device)


    def forward(self, img):
        # img - rgb observation, bev - ground truth bev observation
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0,3,1,2))
        image_val = torch.tensor(img).to(device) / 255.0

        with torch.no_grad():
            # encode rgb image
            image_embed = self.fpvencoder(image_val)

            # add an additional class for ground truth bev
            # dists = torch.cdist(image_embed, torch.cat((bev_lr, self.anchors_lr)))[0]

            # measure 8 classes
            sims = []
            for lr in self.anchors_lr:
                sim = nn.functional.cosine_similarity(image_embed, lr)[0]
                sims.append(sim)
            sims = torch.tensor(sims)
            #probs = nn.functional.softmax(sims)
            y = torch.argmax(sims)
        return y


def readSim():
    cnt = 0

    for i in range(1, 101):
        rgb = cv2.imread(os.path.join("/lab/kiran/img2cmd/test", str(i) + '00.jpg'))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        RGB_img = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)
        id = encoder(RGB_img)
        if id+1 == i:
            cnt += 1
        print(str(i) + '00.jpg', id.cpu().numpy()+1)


    print("accuracy:", cnt / 100)

if __name__ == "__main__":
    root_dir = "test"
    encoder = Encoder("/lab/kiran/ckpts/pretrained/carla/FPV_BEV_CARLA_RANDOM_FPVBEV_CARLA_STANDARD_0.1_0.01_128_512.pt")
    readSim()
    encoder = Encoder("/lab/kiran/ckpts/pretrained/carla/FPV_BEV_CARLA_RANDOM_FPVBEV_CARLA_STANDARD_0.01_0.01_128_512.pt")
    readSim()

