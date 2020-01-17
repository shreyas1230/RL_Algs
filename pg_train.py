from policy_gradient import *
from utils import *
import numpy as np
import torch
import gym

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env_name = 'CartPole-v1'
env = gym.make(env_name)

if len(env.observation_space.shape) == 0:
    input_dim = 1
else:
    input_dim = env.observation_space.shape[0]
if len(env.action_space.shape) == 0:
    output_dim = 1
else:
    output_dim = env.action_space.shape[0]
# print(input_dim, output_dim)
net = PG(input_dim, [10], 2, 0.001).to(device) #Cartpole trainer
net.train(1000, device, env)
collect_trajectories(env, 10, net, 1000, causality=True, baselines=False)
# run_policy(env, 10, net, 1000)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init
