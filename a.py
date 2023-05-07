import gym
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm

my_restored_policy = Policy.from_checkpoint("./testatari/atari_checkpoint/")
print(my_restored_policy.get_weights())
