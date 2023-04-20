import sys
from PIL import Image
import yaml
import random
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
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
#from Vo import FullyConnectedNetwork as TorchFC
#from ray.rllib.models.torch.visionnet import VisionNetwork as TorchFC
from models.AtariModels import VaeNetwork as TorchVae
from models.AtariModels import PreTrainedResNetwork as TorchPreTrainedRes
from models.AtariModels import ResNetwork as TorchRes
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
        "--model",
        choices=["vae", "res", 'pretrainedres', "random", "imagenet", "voltron", "r3m", "value"],
        default="vae",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    
    parser.add_argument(
        "--machine", type=str, default="None", help="machine to be training"
    )
    parser.add_argument(
        "--config_file", type=str, default="/lab/kiran/BeoEnv/hostfile.yaml", help="config file for resources"
    )
    parser.add_argument(
        "--env_name", type=str, default="ALE/Pong-v5", help="ALE/Pong-v5"
    )

    parser.add_argument(
        "--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train."
    )

    parser.add_argument(
        "--num_workers", type=int, default=20, help="Number of GPUs each worker has"
    )
    
    parser.add_argument(
        "--num_envs", type=int, default=8, help="Number of envs each worker evaluates"
    )

    parser.add_argument(
        "--roll_frags", type=int, default=100, help="Rollout fragments"
    )
    
    parser.add_argument(
        "--num_gpus", type=float, default=1, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--gpus_worker", type=float, default=.3, help="Number of GPUs each worker has"
    ) 

    parser.add_argument(
        "--cpus_worker", type=float, default=.5, help="Number of CPUs each worker has"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run with/without Tune using a manual train loop instead. If ran without tune, use PPO without grid search and no TensorBoard.",
    )

    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )


    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config


    class TorchVaeModel(TorchModelV2, nn.Module):

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchVae(
                obs_space, action_space, num_outputs, model_config, name
            )

        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])

    class PreTrainedTorchResModel(TorchModelV2, nn.Module):

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchPreTrainedRes(
                obs_space, action_space, num_outputs, model_config, name
            )

        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])

    class TorchResModel(TorchModelV2, nn.Module):

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchRes(
                obs_space, action_space, num_outputs, model_config, name
            )

        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])


    args = parser.parse_args()
    
    if args.tune:
        args.config_file = '/lab/kiran/BeoEnv/tune.yaml'

    #extract data from the config file
    if args.machine is not None:
        with open(args.config_file, 'r') as cfile:
            config_data = yaml.safe_load(cfile)

    args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
    
    ray.init(local_mode=args.local_mode)

    if args.model=='vae':
        ModelCatalog.register_custom_model(
            "my_model", TorchVaeModel
        )
    elif args.model=='pretrainedres':
        ModelCatalog.register_custom_model(
            "my_model", TorchPreTrainedResModel
        )
    elif args.model=='res':
        ModelCatalog.register_custom_model(
            "my_model", TorchResModel
        )

    config = (
        get_trainable_cls(args.run)
            .get_default_config()
            .environment(args.env_name, clip_rewards = True)
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_workers,
                      rollout_fragment_length = args.roll_frags,
                      num_envs_per_worker = args.num_envs)
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
            .resources(num_gpus=args.num_gpus, num_gpus_per_worker = args.gpus_worker, num_cpus_per_worker=args.cpus_worker
        )
    )



    stop = {
        "timesteps_total": args.stop_timesteps
    }

    if args.tune == False:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        #config.lr = 5e-4
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(10000000):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps:
                break
        algo.stop()
    else:

        hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        }

        pbt = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore,
        )


        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=pbt,
                num_samples = 2,
            ),
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

 
