import time,yaml
from ray.tune.schedulers import PopulationBasedTraining
import networkx as nx
import sys
from PIL import Image
from agent import Agent
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from MplCanvas import MplCanvas
from data_helper import dataHelper, coord_to_sect, coord_to_filename
import numpy as np
import config as app_config
import math, cv2, h5py, argparse, csv, copy, time, os, shutil
from pathlib import Path
import argparse
import ray
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from Zero import ZeroNetwork as TorchZero
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls


class BeoGym(gym.Env):
    
    def __init__(self, config=None):
        config = config or {}

        self.no_image=config.get("no_image",True)
        turning_range = 30 if app_config.PANO_IMAGE_MODE else 60

        super(BeoGym, self).__init__()

        self.dh = dataHelper("data/test.csv", app_config.PANO_HOV)
        self.agent = Agent(self.dh, turning_range, app_config.PANO_IMAGE_RESOLUTION, app_config.PANO_IMAGE_MODE)

        self.action_space = spaces.Discrete(5)
        if self.no_image:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict({"obs": spaces.Box(low = 0, high = 255, shape = (208, 416, 3), dtype= np.uint8), "aux": spaces.Box(low = -1.0, high = 1.0,shape = (5,), dtype= np.float32)})

        self.seed(1)

        # Settings:
        self.game = 'courier'
        self.max_steps = 1000
        self.curr_step = 0
        self.min_radius_meters = 500 # The radius around the goal that can be considered the goal itself.
        self.max_radius_meters = 2000 # The outer radius around the goal where the rewards kicks in.
        self.min_distance_reached = 15 # The closest distance the agent has been so far to the goal.
        self.goal_reward = 100
        self.courier_goal = (66.20711657663878, -17.83818898981032)
        self.last_action = -1
        self.this_action = -1

        # random goal
        # while True:
        #     self.courier_goal = self.dh.sample_location()
        #     self.initial_distance_to_goal = self.dh.distance_to_goal(self.agent.agent_pos_curr, self.courier_goal)
        #     # Make sure the goal is not within the max_radius_meters to the agent's current position. Also, the sampled goal
        #     # should not be the same as the agent's current position:
        #     if (self.initial_distance_to_goal > self.max_radius_meters and self.courier_goal != self.agent.agent_pos_curr):
        #         break
        #     print("Goal is None. Sampled a location as goal: ", self.courier_goal)
        print('Goal is'+str(self.courier_goal))

        # Logging: to be implemented.

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}
        self.curr_step=0
        temp = self.agent.reset()
        aux=[2*(self.agent.agent_pos_curr[0] +100)/200 - 1,2*(self.agent.agent_pos_curr[1] +100)/200 - 1, self.agent.curr_angle/360,2*(self.courier_goal[0] +100)/200 - 1,2*(self.courier_goal[1] +100)/200 - 1]
        self.agent.dis_to_goal = nx.shortest_path_length(self.dh.G, source=self.agent.agent_pos_curr, target=self.courier_goal, weight='weight')
        if self.no_image:
            self.agent.reset()
            return np.array(aux), info
        else:
            return {'obs': temp, 'aux': np.array(aux)}, info

    def step(self, action):
        done = False
        info = {}
        # print("Step with action: ", action)
        self.agent.take_action(action)
        self.last_action=self.this_action
        self.this_action=action
        # Keep track of the number of steps in an episode:
        self.curr_step += 1
        
        if app_config.SAVE_IMAGE_PATH:
            # Image
            action_name = ""
            if action == 0:
                action_name = "straight"
            elif action == 1:
                action_name = "back"
            else:
                angle = self.agent.action_angle_map[action]
                action_name = f"turn_{angle}_degrees"

            filename = f"step_{self.curr_step}_action_{action_name}.{app_config.IMAGE_SOURCE_EXTENSION}"
            cv2.imwrite(f"{app_config.IMAGE_PATH_DIR}/{filename}", self.agent.curr_view)

        if (self.curr_step >= self. max_steps):
            
            done = True
            info['time_limit_reached'] = True

        # print("comparison: ", self.game == 'courier')

        # Three different type of games: https://arxiv.org/pdf/1903.01292.pdf
        if self.game == 'courier':
            reward, terminated = self.courier_reward_fn()
        elif self.game == 'coin':
            reward, terminated = self.coin_reward_fn()
        elif self.game == 'instruction':
            reward, terminated = self.instruction_reward_fn()
        aux = [2*(self.agent.agent_pos_curr[0] +100)/200 - 1,2*(self.agent.agent_pos_curr[1] +100)/200 - 1, self.agent.curr_angle/360,2*(self.courier_goal[0] +100)/200 - 1,2*(self.courier_goal[1] +100)/200 - 1]
        if self.no_image:
            return np.array(aux), reward, terminated, done, info
        else:
            return {'obs': self.agent.curr_view, 'aux': np.array(aux)}, reward, terminated, done, info


    def render(self, mode='human', steps=100):

        # Renders initial plot
        window_name = 'window'
        # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE", self.agent.curr_angle)
        if mode != 'random':
            self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)


        # img = cv2.imshow(self.agent.curr_view)
            cv2.imshow(window_name, self.agent.curr_view)
        # A loop of random actions taken for the amount of steps specified
        if mode == "random":
            self.random_mode(steps, window_name=window_name)

        if mode == "spplanner":
            self.shortest_path_mode(window_name=window_name)

        # This infinite loop is in place to allow keyboard inputs until the program is manually terminated
        while mode == "human":
            key = cv2.waitKeyEx(0)
            # Map keyboard inputs to your actions
            if key == app_config.KeyBoardActions.LEFT:
                print("Left")
                self.agent.go_left()
            elif key == app_config.KeyBoardActions.RIGHT:
                print("Right")
                self.agent.go_right()
            elif key == app_config.KeyBoardActions.FORWARD:
                print("Forward")
                self.agent.go_straight()
            elif key == app_config.KeyBoardActions.REVERSE:
                print("Reverse")
                self.agent.go_back()
            elif key == app_config.KeyBoardActions.KILL_PROGRAM:
                print("Killing Program")
                cv2.destroyAllWindows()
                break

            # If input key is an action related to moving in a certain direction, update the plot based on the action
            # already taken and update the window.

            if key in app_config.DIRECTION_ACTIONS:
                self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
                # img = cv2.imshow(self.agent.curr_view)
                cv2.imshow(window_name, self.agent.curr_view)


            # Update Bird's-Eye-View graph
            graph = self.dh.bird_eye_view(self.agent.agent_pos_curr, r)
            if graph is None:
                raise EnvironmentError("Graph is None")
            self.dh.draw_bird_eye_view(self.agent.agent_pos_curr, r, graph, self.agent.curr_angle)

    def shortest_path_mode(self, window_name=None):
        print("Traversing shortest path to goal")
        shortest_path = self.dh.getShortestPathNodes(self.agent.agent_pos_curr, self.courier_goal)
        print(shortest_path)

        if window_name:
            cv2.imshow(window_name, self.agent.curr_view)
        last_angle = 0
        for i in range(len(shortest_path)):
            angle = 0
            minAngle = 500
            if i != len(shortest_path) - 1:
                agl = self.dh.fix_angle(self.dh.get_angle(shortest_path[i], shortest_path[i + 1]))
                for j in self.dh.camera_angle_map.values():
                    if abs(self.dh.get_distance(agl, j)) < minAngle:
                        angle = j
                        minAngle = abs(self.dh.get_distance(agl, j))
            self.agent.update_agent(shortest_path[i], self.agent.agent_pos_curr, last_angle)
            # self.agent.update_agent(shortest_path[i], self.agent.agent_pos_curr, self.agent.curr_angle)
            self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
            if window_name:
                cv2.imshow(window_name, self.agent.curr_view)
                key = cv2.waitKey(1000)
            while last_angle != angle:
                dis = self.dh.get_distance(last_angle, angle)
                if abs(dis)<=15:
                    last_angle = angle
                else:
                    if abs(self.dh.get_distance(last_angle+15, angle)) < abs(self.dh.get_distance(last_angle-15, angle)):
                        last_angle += 15
                    else:
                        last_angle -= 15
                self.agent.update_agent(self.agent.agent_pos_curr, self.agent.agent_pos_curr, last_angle)
                if window_name:
                    cv2.imshow(window_name, self.agent.curr_view)
                    key = cv2.waitKey(1000)
                last_angle = angle

    def random_mode(self, steps, window_name=None):
        start = time.time()
        for i in range(steps):
            # 7 since there are 6 actions that can be taken. (0-7 since 0 is inclusive and 7 is exclusive )
            self.step(np.random.randint(low=0, high=7))
        end = time.time()
        print(f"----Start: {start}. End: {end}. Difference: {end - start}----")


    # Headless version of the render method
    def render_headless(self, mode="human", steps=0):
        # Renders initial plot
        window_name = 'window'
        # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE", self.agent.curr_angle)
        self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)

        # A loop of random actions taken for the amount of steps specified
        if mode == "random":
            self.random_mode(steps, window_name=None)

        if mode == "spplanner":
            self.shortest_path_mode(window_name=None)

        if mode == "comp":
            action = None
            action_choices = [x for x in range(0, 7)]
            action_choices.append('k')
            # Action is None check is for the initial start of the mode
            while action != 'k' or action is None:
                action = input("Enter an action 0-6, Enter 'k' to kill the program.\n")
                if action == 'k':
                    break
                else:
                    action = int(action)

                if action not in action_choices:
                    print("Received action: ", action)
                    raise EnvironmentError("Action choices must be 0-6 or 'k'.")
                self.step(action)                
                self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
                # time.sleep(2)

        # This infinite loop is in place to allow keyboard inputs until the program is manually terminated
        while mode == "human":
            key = input("Enter operation (Left, Right, Forward, Reverse, Quit): ").lower()

            # Map keyboard inputs to your actions
            if key == app_config.InputActions.LEFT:
                print("Left")
                self.agent.go_left()
            elif key == app_config.InputActions.RIGHT:
                print("Right")
                self.agent.go_right()
            elif key == app_config.InputActions.FORWARD:
                print("Forward")
                self.agent.go_straight()
            elif key == app_config.InputActions.REVERSE:
                print("Reverse")
                self.agent.go_back()
            elif key == app_config.InputActions.KILL_PROGRAM:
                print("Killing Program")
                break

            # If input key is an action related to moving in a certain direction, update the plot based on the action
            # already taken and update the window.
            if key in app_config.DIRECTION_ACTIONS:
                self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)


    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Implementation will be similar to this file: https://github.com/deepmind/streetlearn/blob/master/streetlearn/python/environment/courier_game.py
    def courier_reward_fn(self, distance = None):

        reward = -4
        found_goal = False

        if self.this_action>0:
            return reward,found_goal

        distance_to_goal = nx.shortest_path_length(self.dh.G, source=self.agent.agent_pos_curr, target=self.courier_goal, weight='weight')
        # Does not give reward if the agent visited old locations:
        if self.agent.agent_pos_curr in self.dh.visited_locations:
            self.agent.dis_to_goal = distance_to_goal
            return reward, found_goal
        if distance_to_goal < self.min_radius_meters:
            reward = self.goal_reward
            found_goal = True
        else:
            reward = self.agent.dis_to_goal - distance_to_goal
        self.agent.dis_to_goal = distance_to_goal
        self.dh.visited_locations.add(self.agent.agent_pos_curr)
        return reward, found_goal
        # If distance is not None then we are in testing mode:
        # if distance is None:
        #
        #     distance_to_goal = self.dh.distance_to_goal(self.agent.agent_pos_curr, self.courier_goal)
        #
        #     # Add current location to visited locations list:
        #     self.dh.visited_locations.add(self.agent.agent_pos_curr)
        #
        # else:
        #
        #     distance_to_goal = distance
        # # The goal is reached:
        # if distance_to_goal < self.min_radius_meters:
        #     reward = self.goal_reward
        #     found_goal = True
        # else:
        #     if distance_to_goal < self.max_radius_meters:
        #         # print("max_radius_meters: ", self.max_radius_meters)
        #         # print("min_distance_reached: ", self.min_distance_reached)
        #         # Only rewards the agent if the agent has decreased the closest distance so far to the goal:
        #         if distance_to_goal < self.min_distance_reached:
        #
        #             # Reward is linear function to the distance to goal:
        #             reward = (self.goal_reward *
        #                 (self.max_radius_meters - distance_to_goal) /
        #                 (self.max_radius_meters - self.min_radius_meters))
        #
        #             self.min_distance_reached = distance_to_goal

    def coin_reward_fn(self):
        
        pass

    def instruction_reward_fn(self):
        
        pass


def get_gps_data():
    """
    Read GPS data from a CSV file
    """

    csv_file = f"{app_config.GPS_DATA_PATH}"
    image_data = dict()
    data_image = dict()
    with open(csv_file, newline='') as csvfile:
        gps_reader = csv.reader(csvfile)
        for row in gps_reader:
            # print(row)
            image_name = row[2]
            if image_data.get(image_name):
                raise ValueError("Duplicate images")

            coord_tuple = (row[0], row[1],)
            coord_str = f"{row[0]},{row[1]}"
            image_data[image_name] = coord_tuple
            data_image[coord_str] = image_name
    return image_data, data_image


if __name__ == "__main__":
    # Load the hdf5 files into a global variable
    global coord_to_sect
    pano_mode = app_config.PANO_IMAGE_MODE
    headless_mode = app_config.HEADLESS_MODE
    #mode = app_config.INTERACTION_MODE
    #mode = app_config.ConfigModes.HUMAN
    mode = app_config.ConfigModes.SPPLANNER
    hov_val = app_config.PANO_HOV

    if app_config.SAVE_IMAGE_PATH:
        if not os.path.isabs(app_config.IMAGE_PATH_DIR):
            # Remove directory
            if os.path.exists(app_config.IMAGE_PATH_DIR) and os.path.isdir(app_config.IMAGE_PATH_DIR):
                shutil.rmtree(app_config.IMAGE_PATH_DIR)

            # Create directory
            Path(app_config.IMAGE_PATH_DIR).mkdir(parents=True, exist_ok=True)

    if not pano_mode:
        f = h5py.File("../hd5_files/coordinate_file_map.hdf5")
        for key in f.keys():
            group = f[key]
            values = group.values()
            for dataset in values:
                file_path = dataset[()].decode()
                coord_to_sect[key] = file_path

        gps_map, coord_to_filename_map = get_gps_data()
    csv_file = "data/test.csv"




    r = 128
    steps = 100


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
        "--stop-iters", type=int, default=100, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--machine", type=str, default="None", help="machine to be training"
    )
    parser.add_argument(
        "--config_file", type=str, default="/lab/kiran/BeoEnv/hostfile.yaml", help="config file for resources"
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
    parser.add_argument(
        "--no_image",
        choices=[False,True],
        default=False,
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

    class TorchNoImageModel(TorchModelV2, nn.Module):
        """Example of a PyTorch custom model that just delegates to a fc-net."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchFC(
                obs_space, action_space, num_outputs, model_config, name
            )

        # @profile(precision=5)
        def forward(self, input_dict, state, seq_lens):
            input_dict["obs"] = input_dict["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])

    if headless_mode:
        #unfinished
        env.render_headless(mode=mode, steps=steps)
    else:
        args = parser.parse_args()
        if args.tune:
            args.config_file = '/lab/kiran/BeoEnv/tune.yaml'

            # extract data from the config file
        if args.machine is not None:
            with open(args.config_file, 'r') as cfile:
                config_data = yaml.safe_load(cfile)

        args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
        ray.init(local_mode=args.local_mode)
        if args.no_image:
            ModelCatalog.register_custom_model(
                "my_model", TorchNoImageModel
            )
        else:
            ModelCatalog.register_custom_model(
                "my_model", TorchCustomModel
            )

        config = (
            get_trainable_cls(args.run)
                .get_default_config()
                .environment(BeoGym, env_config = {"no_image":args.no_image}, clip_rewards=True)
                .framework("torch")
                .rollouts(num_rollout_workers=10,
                          rollout_fragment_length='auto',
                          num_envs_per_worker=args.num_envs)
                .training(
                model={
                    "custom_model": "my_model",
                    "vf_share_layers": True,
                    "conv_filters": [[16, [7, 13], 6], [32, [5, 13], 4], [256, [5, 14], 5]],
                    "post_fcnet_hiddens": [128, 64, 64, 32, 32],
                },
                lambda_=0.95,
                kl_coeff=0.5,
                clip_param=0.1,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                train_batch_size=1000,
                sgd_minibatch_size=100,
                num_sgd_iter=10,

            )
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                .resources(num_gpus=args.num_gpus, num_gpus_per_worker=args.gpus_worker,
                           num_cpus_per_worker=args.cpus_worker
                           )
        )



        stop = {
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
        }

        if args.tune == False:
            # manual training with train loop using PPO and fixed learning rate
            if args.run != "PPO":
                raise ValueError("Only support --run PPO with --no-tune.")
            print("Running manual train loop without Ray Tune.")
            # use fixed learning rate instead of grid search (needs tune)
            # config.lr = 5e-4
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
                    num_samples=2,
                ),
                param_space=config.to_dict(),
                run_config=air.RunConfig(stop=stop),
            )
            results = tuner.fit()

            if args.as_test:
                print("Checking if learning goals were achieved")
                check_learning_achieved(results, args.stop_reward)


 
