import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from IPython import embed

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")

class ResNet(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embed_size)

    def forward(self, image):
        out = self.resnet(image)
        return out
