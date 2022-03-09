import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Lin_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Lin_Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, action_dim, bias=False)
        self.max_action = max_action

    def forward(self, x):
        return self.l1(x).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.n1 = nn.LayerNorm(400)
        self.l2 = nn.Linear(400, 300)
        self.n2 = nn.LayerNorm(300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.n4 = nn.LayerNorm(400)
        self.l5 = nn.Linear(400, 300)
        self.n5 = nn.LayerNorm(300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        q1 = self.l1(torch.cat([x, u], 1))
        q1 = self.n1(q1)
        q1 = F.relu(q1)
        q1 = self.l2(q1)
        q1 = self.n2(q1)
        q1 = F.relu(q1)

        q1 = self.l3(q1)

        q2 = self.l4(torch.cat([x, u], 1))
        q2 = self.n4(q2)
        q2 = F.relu(q2)
        q2 = self.l5(q2)
        q2 = self.n5(q2)
        q2 = F.relu(q2)
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, x, u):
        q1 = self.l1(torch.cat([x, u], 1))
        q1 = self.n1(q1)
        q1 = F.relu(q1)
        q1 = self.l2(q1)
        q1 = self.n2(q1)
        q1 = F.relu(q1)

        q1 = self.l3(q1)
        return q1
