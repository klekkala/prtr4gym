import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size, batch_size, num_layers):
        super().__init__()
        self.h = Variable(torch.randn(num_layers, batch_size, hidden_size))
        self.c = Variable(torch.randn(num_layers, batch_size, hidden_size))
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, batch_first=True)

    def forward(self, action, latent):
        seq_lens = [a.size()[0] for a in action]
        in_al = [torch.cat([a, l], dim=-1) for a, l in zip(action,latent)]
        in_padded = torch.nn.utils.rnn.pad_sequence(in_al, batch_first=True)
        in_packed = nn.utils.rnn.pack_padded_sequence(in_padded, seq_lens, batch_first=True, enforce_sorted=False)

        outs, (self.h, self.c) = self.lstm(in_packed, (self.h, self.c))

        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(outs)

        return unpacked