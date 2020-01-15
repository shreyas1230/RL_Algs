from behavior_cloning import *
from utils import *
import numpy as np
import torch
import gym

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env_name = 'Acrobot-v1'
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
net = BC(input_dim, [50,20,3], output_dim, 0.001).to(device)
# expertmodel, data = gen_exp_data(env_name, 50)
# model = trainexpert(env_name, 30)
# testexpert(env, expertmodel, 20, 10000)


data = loadexpertdata('expert.npz')
loader = make_dataloader(torch.from_numpy(np.array(data['obs'])).float(), torch.from_numpy(np.array(data['actions'])).float(), batch_size = 1000)
net.train(1000, loader, device, env)
run_policy(env, 10, net, 1000)
