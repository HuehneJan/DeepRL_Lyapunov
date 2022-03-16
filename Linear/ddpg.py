import torch
import torch.nn.functional as F
from torch.optim import Adam
from nets import *
from torch.optim import lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPG(object):
    def __init__(self, gamma, tau, num_inputs, action_space,):
        """
        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic

            num_inputs:     Size of the input states
            action_space:   The action space of the used environment. Used to clip the actions and
                            to distinguish the number of outputs
        """

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define the actor
        self.actor = Actor(num_inputs, self.action_space).to(device)
        self.actor_target = Actor(num_inputs, self.action_space).to(device)

        # Define the critic
        self.critic = Critic(num_inputs, self.action_space).to(device)
        self.critic_target = Critic(num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-4,
                                    )  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-3,
                                     #weight_decay=0.1
                                     )  # optimizer for the critic network
        # Optional: LR Scheduler for actor and critic network
        self.actor_lr_scheduler = lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.8)
        #self.critic_lr_scheduler = lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.8)

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def calc_action(self, state, action_noise=None):
        """
                Calculates the action to perform in a given state
                Arguments:
                    state:          State to perform the action on in the env.
                                    Used to evaluate the action.
                    action_noise:   The noise generating process to apply on the evaluated action,
                                    if None no noise is applied
        """
        x = state.to(device)

        self.actor.eval()
        mu = self.actor(x)
        self.actor.train()

        mu = mu.data

        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise

        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])
        return mu

    def update_params(self, batch, **kwargs):
        """
                Updates the parameters/networks of the agent according to the given batch.
                Arguments:
                    batch:  Batch to perform the training of the parameters
                    policy_update: If true a policy update will be applied
        """
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        done_batch = done_batch.unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state).to(device)
        with torch.no_grad():
            next_action_batch = self.actor_target(next_state_batch)
            next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())
            targetQ = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, targetQ)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.05, norm_type=2)
        self.actor_optimizer.step()
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def set_eval(self):
        """
        Sets the model in evaluation mode
        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode
        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()