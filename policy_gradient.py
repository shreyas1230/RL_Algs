import torch
from torch.distributions import Categorical
import numpy as np
from utils import run_policy, collect_trajectories, make_dataloader, rew_to_go, cum_rew

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

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.relu(x)
            else:
                x = self.softmax(x)
        return x

    def train(self, num_epochs, device, env, causality = False, baselines = False):
        for epoch in range(num_epochs):
            rend = epoch%100 == 0
            obs, acs, rews, dones, log_probs, baseline = collect_trajectories(env, 20, self, 1000, rend)
            if causality:
                r = rew_to_go(np.array(rews), np.array(dones))
            else:
                r = cum_rew(np.array(rews), np.array(dones))
            adv = r - baseline if baselines else r
            loss = 0
            ind = np.arange(len(log_probs))
            np.random.shuffle(ind)
            for i in ind:
                loss -= adv[i]*log_probs[i]
            print('EPOCH {0}'.format(epoch+1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('Loss: {0}'.format(loss.item()))
            print()

    def sample_action(self, observation):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        action_probs = self(torch.from_numpy(np.array(observation)).float().to(device))
        d = self.dist(action_probs)
        action = d.sample()
        lp = d.log_prob(action)
        return action.cpu().numpy(), lp
