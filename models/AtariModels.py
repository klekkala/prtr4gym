import numpy as np
from typing import Dict, List
import gymnasium as gym
from models.ResnetX import VAE as VAE
from torchvision.models import resnet18, ResNet18_Weights


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

        self._vae=VAE(channel_in=3, ch=64)
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




class FrozenResNetwork(TorchModelV2, nn.Module):
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
        self._resnet = resnet18()
        self._preprocess = weights.transforms()

        self._resnet.train()


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