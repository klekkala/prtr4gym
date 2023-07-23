
import torch
import torch.nn as nn
from torch.autograd import Variable

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size, batch_size, num_layers, vae=None):
        super().__init__()
        self.vae = vae
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, batch_first=True)
        self.h_size = (num_layers, batch_size, hidden_size)
        self.init_hs()

    def init_hs(self):
        self.h_0 = Variable(torch.randn(self.h_size)).to(device)
        self.c_0 = Variable(torch.randn(self.h_size)).to(device)

    def encode(self, image):
        x = torch.reshape(image, (-1,) + image.shape[-3:])
        _, mu, logvar = self.vae(x)
        z = self.vae.reparameterize(mu, logvar)
        z = torch.reshape(z, image.shape[:2] + (-1,))
        return z, mu, logvar

    def decode(self, z):
        z_f = torch.reshape(z,  (-1,) + (z.shape[-1],))
        img = self.vae.recon(z_f)
        return torch.reshape(img, z.shape[:2] + img.shape[-3:])

    def forward(self, action, latent):
        print(latent.shape)
        in_al = torch.cat([torch.Tensor(action), latent], dim=-1)
        outs, _ = self.lstm(in_al.float(), (self.h_0, self.c_0))
        return outs
'''
class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)

        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)

        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs
'''
