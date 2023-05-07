# Create a rollout worker and using it to collect experiences.
import gym
import numpy as np
import cv2
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
from ray.rllib.algorithms.algorithm import Algorithm
my_restored_policy = Policy.from_checkpoint("./testatari/atari_checkpoint/")

#print(PGTF1Policy)
#print(my_restored_policy)

#worker = RolloutWorker( 
#  env_creator=lambda _: gym.make("CartPole-v1"),
#  policy_spec=PGTF1Policy)
#worker.add_policy(my_restored_policy)
class MultiTaskEnv(gym.Env):
        def __init__(self, env_config):
            self.env = gym.make("NameThisGameNoFrameskip-v4", full_action_space=True)
            self.name= "NameThisGameNoFrameskip-v4"
            #self.env = wrap_deepmind(self.env)
            self.action_space = self.env.action_space
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8) #self.env.observation_space
            #if self.observation_space.shape[0]==214:
                #self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)

        def reset(self):
            temp = self.env.reset()
            if isinstance(temp, np.ndarray):
                return cv2.resize(temp, (84, 84))
            #if str(type(temp))!='tuple':
                #return cv2.resize(temp, (84, 84))
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1]))
            return tuple(temp)

        def step(self, action):
            temp = self.env.step(action)
            if isinstance(temp, np.ndarray):
                return cv2.resize(temp, (84, 84))
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1],temp[2],temp[3],temp[4]))
            return tuple(temp)

config = {
    "env" : MultiTaskEnv,
    "clip_rewards" : True,
    "framework" : "torch",
    "model":{
            "vf_share_layers" : True,
    }
    }
algo = PPO(config)

#algo.train()
#path_to_checkpoint = algo.save()
#print(
#    "An Algorithm checkpoint has been created inside directory: "
#    f"'{path_to_checkpoint}'."
#)
#algo.stop()
#algo.remove_policy(policy_id='default_policy')
#algo.add_policy(policy_id='default_policy',policy=my_restored_policy)

path_to_checkpoint = '/lab/kiran/ray_results/PPO_MultiTaskEnv_2023-05-02_15-59-30gwrctica/checkpoint_000001'
algo = Algorithm.from_checkpoint(path_to_checkpoint)
algo.set_weights(my_restored_policy.get_weights())
algo.evaluate()
