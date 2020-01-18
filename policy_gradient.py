import torch
from torch.distributions import Categorical
import numpy as np
from utils import run_policy, collect_trajectories, make_dataloader, rew_to_go, cum_rew
from replay_buffer import *

class PG(torch.nn.Module):
    def __init__(self, D_in, H, D_out, lr=0.01, dist=Categorical):
        super(PG, self).__init__()
        dims = []
        dims.append(D_in)
        dims.extend(H)
        dims.append(D_out)
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.dist = dist
        self.mem = ReplayMemory(100000)
        self.input_dim = D_in
        self.output_dim = D_out

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.relu(x)
            else:
                x = self.softmax(x)
        return x

    def train(self, env, num_epochs, batch_size, device, causality = False, baselines = False):
        for epoch in range(num_epochs):
            print('EPOCH {0}'.format(epoch+1))
            rend = epoch % 100 == 0
            # obs, acs, rews, dones, log_probs, baseline = collect_trajectories(env, 20, self, 1000, self.mem)
            collect_trajectories(env, 1, self, 1000, self.mem, device, rend)
            transitions = self.mem.sample_recent(batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).reshape((-1, self.input_dim))
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_state_batch = torch.cat(batch.next_state).reshape((-1, self.input_dim))
            done_batch = torch.cat(batch.done)
            baseline = torch.sum(reward_batch)/torch.sum(done_batch)

            if causality:
                r = rew_to_go(reward_batch, done_batch)
            else:
                r = cum_rew(reward_batch, done_batch)
            adv = r - baseline if baselines else r

            action_probs = self(state_batch)
            log_probs = self.dist(action_probs).log_prob(action_batch)
            loss = torch.mean(-log_probs*r)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('Loss: {0}'.format(loss.item()))
            print()

    def sample_action(self, observation):
        action_probs = self(observation)
        d = self.dist(action_probs)
        action = d.sample().view(1,1)[0]
        return action
