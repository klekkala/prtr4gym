import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


class VAEBEV(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(VAEBEV, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z)
        self.fc2 = nn.Linear(h_dim, z)
        self.fc3 = nn.Linear(z, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        i = torch.reshape(x, (-1,) + x.shape[-3:])
        z = self.bottleneck(self.encoder(i))[0]
        return torch.reshape(z, x.shape[:2] + (-1,))

    def forward(self, x):
        i = torch.reshape(x, (-1,) + x.shape[-3:])
        h = self.encoder(i)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        o = self.decoder(z)
        o = torch.reshape(o, x.shape)
        return o, mu, logvar

        
class VAE(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),

        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)
        self.conv_log_var = nn.Conv2d(ch*32, z, 1, 1)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
        )
        
    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

        
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.conv_mu(h), self.conv_log_var(h)
        mu = torch.flatten(mu, start_dim=1)
        log_var = torch.flatten(log_var, start_dim=1)
        encoding = self.sample(mu, log_var)
        encoding = mu
        return self.decoder(encoding), mu, log_var
        #return mu



class Encoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),

        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)


        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.conv_mu(h)
        mu = torch.flatten(mu, start_dim=1)
        return mu


class TEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(TEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),

        )


        
    def forward(self, x):
        h = self.encoder(x)
        mu = torch.flatten(h, start_dim=1)
        return mu
