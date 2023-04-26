import argparse
import os
# workaround to unpickle olf model files
import sys
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from Zero import ZeroNetwork as TorchZero
import gymnasium as gym
import numpy as np
import torch
from gym.envs.registration import register
from ray.rllib.policy.policy import Policy
import cv2
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env-name',
    default='beogym-v0',
    help='environment to train on')
args = parser.parse_args()


env = gym.make(args.env_name)
# my_new_ppo = Algorithm.from_checkpoint('C:/Users/b5438/ray_results/PPO_BeoGym_2023-04-22_21-14-28q8q1jzr4/checkpoint_000020')

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchZero(
            obs_space, action_space, num_outputs, model_config, name
        )

    # @profile(precision=5)
    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


ModelCatalog.register_custom_model("my_model", TorchCustomModel)

my_restored_policy = Policy.from_checkpoint("/tmp/my_policy_checkpoint")

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
obs, info = env.reset()

count = 0
while True:
    count += 1
    action = my_restored_policy.compute_single_action(obs)[0]
    # Obser reward and next obs
    obs, reward, terminated, done, _ = env.step(action)
    env.dh.update_plot(env.agent.agent_pos_curr, env.courier_goal)
    cv2.imshow('window', obs['obs'])
    key = cv2.waitKey(100)
    print("reward is:"+str(reward))
    if count==100:
        break