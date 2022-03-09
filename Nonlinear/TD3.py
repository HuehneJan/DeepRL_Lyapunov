import torch
import torch.nn.functional as F
from networks import Actor, Critic, Lin_Actor
from utils import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TD3(object):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 batch_size=64,
                 gamma=0.99,
                 tau=0.01,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=10,
                 use_linear=False
                 ):
        """
        Creates an TD3 Object than can be used for TD3-Algorithm training.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param max_action: Maximum absolute control input value.
        :param batch_size: Batch size used for neural network training.
        :param gamma: Discount factor used.
        :param tau: Target Network update parameter.
        :param policy_noise: Policy noise standard deviation
        :param noise_clip: Policy noise maximum absolute value.
        :param policy_freq: Policy update frequency.
        :param use_linear: Parameter to set used actor to linear NN actor. Defaults to Nonlinear NN Actor
        """
        if use_linear:
            self.actor = Lin_Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target = Lin_Actor(state_dim, action_dim, max_action).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
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
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.updated = 0

    def calc_action(self, state, noise=None):
        """
        Method to create next action with noise exploration if required.
        :param state: Torch Tensor state for which the action will be calculated with self.actor
        :param noise: noise generation function object that should be used. If  specified noise will be applied,
        else no noise will be applied. Has to contain a method noise() that returns a noise value if called.
        :return: torch.Tensor action+noise
        """
        if noise is None:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            noise = torch.FloatTensor(noise.noise()).to(device)
            action = (self.actor(state) + noise).clamp(-self.max_action, self.max_action)
            return action.cpu().data.numpy().flatten()

    def update(self):
        self.updated += 1
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            target_Q1, target_Q2 = self.critic_target(next_state,
                                                      (self.actor_target(next_state)
                                                       + noise).clamp(-self.max_action,self.max_action))
            target_Q = reward + (not_done * self.gamma + torch.min(target_Q1, target_Q2)).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.updated % self.policy_freq == 0:
            self.critic.eval()
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.05, norm_type=2)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.cpu().item()
