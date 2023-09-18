from models.atari_vae import VAEBEV
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
from IPython import embed
import cv2
import os
import numpy as np
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")



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
    def __init__(self, encoder_path, classification=False):
        super().__init__()
        self.fpvencoder = ResNet(32).to(device)

        # vaeencoder
        self.bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        self.bevencoder.load_state_dict(vae_ckpt['model_state_dict'])
        self.bevencoder.eval()
        for param in self.bevencoder.parameters():
            param.requires_grad = False

        # load models
        checkpoint = torch.load(encoder_path, map_location="cpu")
        print(checkpoint['epoch'])
        self.fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
        self.fpvencoder.eval()
        for param in self.fpvencoder.parameters():
            param.requires_grad = False


        # read anchor images and convert to latent representations
        self.anchors_lr = []
        self.anchors = []
        self.label = []
        if not classification:
            root = "/home/carla/img2cmd/test"
        else:
            root = "/lab/kiran/img2cmd/anchor"

        for root, subdirs, files in os.walk(root):
            if files:
                for f in files:
                    if '.jpg' in f:
                        im = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                        self.anchors.append(im)
                        with torch.no_grad():
                            im = np.expand_dims(im, axis=(0, 1))
                            im = torch.tensor(im).to(device) / 255.0
                            _, embed_mu, embed_logvar = self.bevencoder(im)
                            embed = embed_mu.cpu().numpy()[0]
                            self.anchors_lr.append(embed)
        self.anchors = np.array(self.anchors)
        self.anchors_lr = np.array(self.anchors_lr)
        self.anchors_lr = torch.tensor(self.anchors_lr).to(device)


    def forward(self, img):
        image_val = img.to(device) / 255.0
        with torch.no_grad():
            _, image_embed, _ = self.bevencoder(image_val)
        # compare embeddings
        #sims = nn.functional.cosine_similarity(image_embed, self.anchors_lr)
        sims = f.normalize(image_embed) @ f.normalize(self.anchors_lr).t()

        ys = torch.argmax(sims, axis=1)

        return ys.cpu().numpy(), torch.max(sims, axis=1), image_embed


class StateLSTM(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, encoder):
        super().__init__()
        self.encoder = encoder
        if encoder is not None:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True).cuda()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_hs(self, batch_size):
        self.h_0 = Variable(torch.randn((self.num_layers, batch_size, self.hidden_size))).to(device)
        self.c_0 = Variable(torch.randn((self.num_layers, batch_size, self.hidden_size))).to(device)

    def forward(self, image):
        #x = torch.reshape(image, (-1,) + image.shape[-3:]).float()
        x = image
        z = self.encoder(x).float()
        z = torch.reshape(z, (1, image.shape[0], -1))
        #z = torch.reshape(z, image.shape[:2] + (-1,))
        outs, (self.h_0, self.c_0) = self.lstm(z.float(), (self.h_0, self.c_0))
        return outs

        


class StateActionLSTM(StateLSTM):
    def __init__(self, latent_size, action_size, hidden_size, num_layers, encoder=None, vae=None):
        super().__init__(latent_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, encoder=encoder)
        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, num_layers, batch_first=True)
        self.encoder = Encoder("/lab/kiran/ckpts/pretrained/carla/FPV_BEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt", True)


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
        in_al = torch.cat([action, latent], dim=-1)
        outs, (self.h_0, self.c_0) = self.lstm(in_al.float(), (self.h_0, self.c_0))
        return outs


class MDLSTM(StateActionLSTM):
    def __init__(self, latent_size, action_size, hidden_size, num_layers, gaussian_size, encoder=None, vae=None):
        super().__init__(latent_size, action_size, hidden_size, num_layers, encoder, vae)
        self.gaussian_size = gaussian_size
        self.gmm_linear = nn.Linear(hidden_size, (2 * latent_size + 1) * gaussian_size)

    def forward(self, action, latent):
        seq_len, bs = action.size(0), action.size(1)
        in_al = torch.cat([torch.Tensor(action), latent], dim=-1)
        outs, (self.h_0, self.c_0) = self.lstm(in_al.float(), (self.h_0, self.c_0))

        gmm_outs = self.gmm_linear(outs)
        stride = self.gaussian_size * self.latent_size

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussian_size, self.latent_size)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussian_size, self.latent_size)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussian_size]
        pi = pi.view(seq_len, bs, self.gaussian_size)
        logpi = f.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi
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


