import sys
from PIL import Image


import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path

import argparse
import ray
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
#from Vo import FullyConnectedNetwork as TorchFC
#from ray.rllib.models.torch.visionnet import VisionNetwork as TorchFC
from Atari101Model import VaeNetwork as TorchZero
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
#from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.evaluation import evaluate_policy

#from stable_baselines3 import PPO, A2C



if __name__ == "__main__":
    # Load the hdf5 files into a global variable



    torch, nn = try_import_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--framework",
        choices=["torch"],
        default="torch",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=50, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. In this case,"
             "use PPO without grid search and no TensorBoard.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )



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

        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])


    args = parser.parse_args()
    ray.init(local_mode=args.local_mode)
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
    )

    config = (
        get_trainable_cls(args.run)
            .get_default_config()
            .environment("ALE/Pong-v5", clip_rewards = True)
            .framework(args.framework)
            .rollouts(num_rollout_workers=8,
                      rollout_fragment_length= 'auto',
                      num_envs_per_worker = 6)
            .training(
            model={
                "custom_model": "my_model",
                "vf_share_layers": True,
            },
            lambda_ = 0.95,
            kl_coeff = 0.5,
            clip_param = 0.1,
            vf_clip_param = 10.0,
            entropy_coeff = 0.01,
            train_batch_size=5000,
            sgd_minibatch_size=500,
            num_sgd_iter=10,

        )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=1,num_gpus_per_worker = 2, num_cpus_per_worker=12
    )
    )
    # config = {
    #     "framework":torch,
    #     "num_envs_per_worker": 1,
    #     "rollout_fragment_length": 1000,
    #     "train_batch_size": 10000,
    #     "learning_rate": 0.0001,
    #     "clip_param": 0.1,
    #     "clip_rewards": True,
    #     "num_sgd_iter": 10,
    #     "num_workers": 8,
    #     "num_gpus" : 1,
    #     ""
    #     "lambda": 0.95,
    #     "kl_coeff": 0.5,
    # }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                    result["timesteps_total"] >= args.stop_timesteps
                    or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    env.close()
    
 
