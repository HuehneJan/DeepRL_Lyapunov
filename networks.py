import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400, bias=False)
        self.l2 = nn.Linear(400, 300, bias=False)
        self.l3 = nn.Linear(300, action_dim, bias=False)

        self.max_action = max_action

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x
class Lin_Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(Lin_Actor, self).__init__()
            self.l1 = nn.Linear(state_dim, action_dim, bias=False)
            self.max_action = max_action
        def forward(self, x):
            return self.l1(x).clamp( -self.max_action, self.max_action)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300,1)

    def forward(self,x,u):
        x = F.relu(self.l1(torch.cat([x,u],1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

