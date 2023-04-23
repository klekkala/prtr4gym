import numpy as np
from typing import Dict, List
#import gymnasium as gym
import gym
from models.ResnetX import VAE as VAE
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torchvision.transforms as transforms




torch, nn = try_import_torch()


class VaeNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):



        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)



        self._logits = None

        self._vae=VAE(channel_in=4, ch=64)
        checkpoint = torch.load("/lab/kiran/models/pretrained/atari/" + "STL10_ATTARI_64.pt", map_location="cpu")
        print("Checkpoint loaded")
        self._vae.load_state_dict(checkpoint['model_state_dict'])


        layers=[]

        in_size = 512

        layers.append(
            SlimFC(
                in_size=in_size,
                out_size=num_outputs,
                activation_fn=None,
                initializer=normc_initializer(1.0),
            )
        )

        self._logits = layers.pop()

        self._value_branch = SlimFC(
            in_size, 1, initializer=normc_initializer(0.01), activation_fn=None
        )


        for name, param in self._vae.named_parameters():
            param.requires_grad = False


        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        self._resize_transform = transforms.Resize((84, 84))
        self._features = self._resize_transform(self._features)
        mod_x = self._vae(self._features / 255.0)[1].detach()
        vae_out = mod_x.view(self._features.shape[0], -1)
        self._features = vae_out
        vae_out = self._logits(vae_out)
        return vae_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)





class PreTrainedResNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):



        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)



        self._logits = None
        weights = ResNet18_Weights.IMAGENET1K_V1
        self._resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._preprocess = weights.transforms()

        self._resnet.eval()


        layers=[]

        in_size = 1000

        layers.append(
            SlimFC(
                in_size=in_size,
                out_size=num_outputs,
                activation_fn=None,
                initializer=normc_initializer(1.0),
            )
        )

        self._logits = layers.pop()

        self._value_branch = SlimFC(
            in_size, 1, initializer=normc_initializer(0.01), activation_fn=None
        )


        for name, param in self._resnet.named_parameters():
            param.requires_grad = False


        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        res_out = self._preprocess(self._features)
        res_out = self._resnet(res_out)
        self._features = res_out
        res_out = self._logits(res_out)
        return res_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


class ResX(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """

    def __init__(self, channels, ch=64, z=512):
        super(ResX, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2 * ch)  # 32
        self.conv3 = ResDown(2 * ch, 4 * ch)  # 16
        self.conv4 = ResDown(4 * ch, 8 * ch)  # 8
        self.conv5 = ResDown(8 * ch, 8 * ch)  # 4
        self.conv_mu = nn.Conv2d(8 * ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8 * ch, z, 2, 2)  # 2

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.training:
            mu = self.conv_mu(x)
            log_var = self.conv_log_var(x)
            x = self.sample(mu, log_var)
        else:
            mu = self.conv_mu(x)
            x = mu
            log_var = None

        return x, mu, log_var


class ResNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):



        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)



        self._logits = None
        self._encoder=ResX(4, 64, 512)
        self._encoder.train()


        layers=[]

        in_size = 512

        layers.append(
            SlimFC(
                in_size=in_size,
                out_size=num_outputs,
                activation_fn=None,
                initializer=normc_initializer(1.0),
            )
        )

        self._logits = layers.pop()

        self._value_branch = SlimFC(
            in_size, 1, initializer=normc_initializer(0.01), activation_fn=None
        )


        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._encoder.train()
        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        self._resize_transform = transforms.Resize((84, 84))
        self._features = self._resize_transform(self._features)
        res_out = self._encoder(self._features / 255.0)[1].detach()
        res_out = res_out.view(self._features.shape[0], -1)
        self._features = res_out
        res_out = self._logits(res_out)
        return res_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)
