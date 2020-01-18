import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gym
from stable_baselines import A2C
from stable_baselines.gail import ExpertDataset, generate_expert_traj

class Data(Dataset):
  def __init__(self,x,y):
      self.x = x#torch.from_numpy(np.array(x)).float()
      self.y = y#torch.from_numpy(np.array(y)).float()
      print(self.x.dtype)
      self.len = self.x.shape[0]
  def __len__(self):
      return self.len
  def __getitem__(self,index):
      return self.x[index], self.y[index]

def make_dataloader(x, y, batch_size = 100):
    data = Data(x, y)
    loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    return loader

def collect_trajectories(env, epochs, policy, time_per_epoch, memory, device, rend):
    timesteps = time_per_epoch
    trajrew = []
    for epoch in range(epochs):
        observation = env.reset()
        totrew = 0
        for t in range(timesteps):
            if rend:
                env.render()
            observation = torch.from_numpy(observation).float().to(device)
            action = policy.sample_action(observation)
            new_obs, reward, done, info = env.step(action.item())
            new_observation = torch.from_numpy(new_obs).float().to(device)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([1 if done or t==timesteps-1 else 0], device=device)
            memory.push(observation, action, new_observation, reward, done)
            observation = new_obs
            totrew+=reward
            if done.item():
                # print("Episode finished after {} timesteps".format(t+1))
                break
        trajrew.append(totrew)
    if rend:
        env.close()
    print("Mean Reward Per Episode: {0}".format((sum(trajrew)/len(trajrew)).item()))

def run_policy(env, epochs, policy, time_per_epoch):
    rews = []
    timesteps = time_per_epoch
    for epochs in range(epochs):
        totrew = 0
        observation = env.reset()
        for t in range(timesteps):
            env.render()
            with torch.no_grad():
                action = policy(torch.from_numpy(np.array(observation)).float()).numpy()
                if action.shape[0] == 1:
                    action = int(round(action[0]))
            observation, reward, done, info = env.step(action)
            totrew += reward
            if done or t==timesteps-1:
                rews.append(totrew)
                print("Episode finished after {} timesteps".format(t+1))
                print("Total Reward: {0}\n".format(totrew))
                break
    env.close()
    print("Mean Reward: {0}".format(sum(rews)*1.0/len(rews)))

def gen_exp_data(env_name, epochs):
    model = trainexpert(env_name, epochs)
    return model, loadexpertdata('expert.npz')

def trainexpert(env_name, epochs):
    model = A2C('MlpPolicy', env_name, verbose=1)
    generate_expert_traj(model, 'expert', n_timesteps=int(1e6), n_episodes=epochs)  #expert data saved in file named 'expert.npz'
    return model

def testexpert(env, model, num_epochs, time_per_epoch):
    reward_sum = 0.0
    timesteps = time_per_epoch
    for epochs in range(num_epochs):
        observation = env.reset()
        for t in range(timesteps):
            env.render()
            # print(observation)
            with torch.no_grad():
                action,_ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            if done or t==timesteps-1:
                print("Episode finished after {} timesteps".format(t+1))
                print(reward_sum)
                reward_sum = 0.0
                break
    env.close()

def loadexpertdata(path):
    return np.load(path)

def rew_to_go(rews, dones):
    start, end = 0, 0
    trajrew = []
    for i in dones.nonzero():
        end = i+1
        trajrew.append(torch.flip(torch.cumsum(rews[start:end], dim=0), dims=[0]))
        start = end
    return torch.cat(trajrew)

def cum_rew(rews, dones):
    start, end = 0, 0
    trajrew = []
    for i in dones.nonzero():
        end = i+1
        trajrew.append(torch.sum(rews[start:end]).repeat_interleave(end-start))
        start = end
    return torch.cat(trajrew)
