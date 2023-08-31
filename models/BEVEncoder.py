import torch
import torch.nn as nn
from IPython import embed

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
    
