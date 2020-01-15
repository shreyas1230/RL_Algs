import torch
import numpy as np
from utils import run_policy

class BC(torch.nn.Module):
    def __init__(self, D_in, H, D_out, lr=0.01):
        super(BC, self).__init__()
        dims = []
        dims.append(D_in)
        dims.extend(H)
        dims.append(D_out)
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.relu(x)
            else:
                x = self.sigmoid(x)
        return x

    def train(self, num_epochs, data, device, env):
        for epoch in range(num_epochs):
            print('EPOCH {0}'.format(epoch+1))
            totalloss = 0
            for batch_idx, (x,y) in enumerate(data):
                x,y = x.to(device), y.to(device)
                prediction = self(x)
                loss = self.criterion(prediction, y)
                totalloss+=loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('Loss: {0}'.format(totalloss))
            if epoch%50 == 0:
                run_policy(env, 10, self, 1000)
            print()
