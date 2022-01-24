import gym
import torch
import numpy as np
import DDPG
from torch.utils.tensorboard import SummaryWriter
from noise import OrnsteinUhlenbeckActionNoise

if __name__ == "__main__":

    writer = SummaryWriter("./lin_runs/run2")
    env = gym.make("Pendulum-v1")

    epochs = 30000
    tau = 0.001
    gamma = 0.99
    noise_stddev = 0.2
    batch_size = 64
    max_epoch_length = 300
    action_dim = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim),
                                            sigma=float(noise_stddev) * np.ones(action_dim))
    state_dim = env.observation_space.shape[-1]
    max_action = env.action_space.high[0]
    count = 0
    agent = DDPG.DDPG(state_dim=state_dim, action_dim=action_dim, max_action=max_action, gamma=gamma, tau=tau)
    for epoch in range(epochs):
        state, done = env.reset(), False
        ou_noise.reset()
        ep_r = 0
        ep_step = 0
        ep_loss = 0
        while(True):
            ep_step += 1
            count += 1
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action)
            done_float = float(done)
            ep_r += reward
            agent.replay_buffer.add(state, action, next_state, reward, done_float)
            state = next_state
            if count > batch_size:
                value_loss, policy_loss = agent.update()
                ep_loss += value_loss
            if done or ep_step > 200:
                break
            if abs(env.state[0]) > 0.2:
                break
        weights = agent.actor.l1.weight.cpu().data.numpy()
        writer.add_scalar("epoch/value_loss", ep_loss/ep_step, epoch)
        writer.add_scalar("epoch/return", ep_r, epoch)
        writer.add_scalar("epoch/k1", weights[0,0], epoch)
        writer.add_scalar("epoch/k2", weights[0, 1], epoch)

        print(f"Epoch: {epoch}  Loss {ep_loss / ep_step}   Reward: {ep_r} Epoch Steps {ep_step} ")
        if epoch % 100 == 0:
            torch.save(agent.actor, "./linear" + "/actor%s.pt" % epoch)
            torch.save(agent.critic, "./linear" + "/critic%s.pt" % epoch)

