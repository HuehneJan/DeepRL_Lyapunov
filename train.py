import argparse
import torch
import numpy as np
import TD3
from custom_env import PendulumEnv
from torch.utils.tensorboard import SummaryWriter
from noise import OrnsteinUhlenbeckActionNoise
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 Training on Nonlinear Pendulum System.')
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--length", type=float, default=1.0, help="Length of pendulum.")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass of pendulum.")
    parser.add_argument("--max_action", type=float, default=2.0, help="Maximum torque.")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise applied to next state of environment")
    parser.add_argument("--use_linear", action="store_true", default=False)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(os.getcwd(), f"data/{args.save_dir}")):
        os.makedirs(os.path.join(os.getcwd(), f"data/{args.save_dir}"), 0o777)
    if not os.path.exists(os.path.join(os.getcwd(), "tensorboard_runs")):
        os.mkdir("./tensorboard_runs")

    env = PendulumEnv(m=args.mass, l=args.length, noise=args.noise, max_action=args.max_action)
    writer = SummaryWriter(f"./tensorboard_runs/{args.run_name}")
    epochs = 40000
    tau = 0.01
    gamma = 0.99
    noise_stddev = 0.2
    batch_size = 64
    max_epoch_length = 200
    action_dim = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim),
                                            sigma=float(noise_stddev) * np.ones(action_dim))
    state_dim = env.observation_space.shape[-1]
    max_action = env.action_space.high[0]
    count = 0
    agent = TD3.TD3(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    gamma=gamma,
                    tau=tau,
                    policy_freq=10,
                    use_linear=args.use_linear
                    )
    epoch_steps, returns, losses = [], [], []
    for epoch in range(epochs):
        state, done = env.reset(), False
        ou_noise.reset()
        ep_r = 0
        ep_step = 0
        ep_loss = 0
        while True:
            ep_step += 1
            count += 1
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action)
            done_float = float(done)
            ep_r += reward
            agent.replay_buffer.add(state, action, next_state, reward, done_float)
            state = next_state
            if count > batch_size:
                value_loss = agent.update()
                ep_loss += value_loss
            if done or ep_step >= max_epoch_length:
                break
        epoch_steps.append(ep_step)
        returns.append(ep_r)
        losses.append(ep_loss/ ep_step)
        writer.add_scalar("epoch/value_loss", ep_loss/ep_step, epoch)
        writer.add_scalar("epoch/return", ep_r, epoch)

        writer.add_scalar("epoch/ep_steps", ep_step, epoch)
        print(f"Epoch: {epoch}  Loss {ep_loss / ep_step}   Reward: {ep_r} Epoch Steps {ep_step} ")
        if epoch % 100 == 0:
            torch.save(agent.actor, f"./data/{args.save_dir}" + "/actor%s.pt" % epoch)
            torch.save(agent.critic, f"./data/{args.save_dir}" + "/critic%s.pt" % epoch)

    epoch_steps = np.array(epoch_steps)
    np.save(f"./data/{args.save_dir}/epoch_steps.npy", epoch_steps)
    returns = np.array(returns)
    np.save(f"./data/{args.save_dir}/returns.npy", returns)
    losses = np.array(losses)
    np.save(f"./data/{args.save_dir}/av_loss.npy", losses)

