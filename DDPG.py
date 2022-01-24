import torch
import torch.nn.functional as F
from networks import Actor, Critic, Lin_Actor
from utils import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, batch_size=64, gamma=0.99, tau = 0.001):
        self.actor = Lin_Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Lin_Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.batch_size = batch_size
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.gamma = gamma
        self.tau = tau

    def calc_action(self, state, noise=None):
        if noise is None:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            noise = torch.FloatTensor(noise.noise()).to(device)
            action = (self.actor(state) + noise).clamp(-self.max_action, self.max_action)
            return action.cpu().data.numpy().flatten()
    def update(self):
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.gamma + target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.eval()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.05, norm_type=2)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.cpu().item(), actor_loss.cpu().item()