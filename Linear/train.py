from TD3 import TD3
from ddpg import *
from replay_memory import *
from noise import *
from torch.utils.tensorboard import SummaryWriter
from custum_env import *
import argparse
from utils import *
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="DDPG_TEST")
    parser.add_argument("--run_name", type=str, default="TEST")
    parser.add_argument("--env", help="Either 1 or 2 for the respective linear system.", default=2, type=int)
    parser.add_argument("--batch_size", help="Batch size for training.", default=64, type=int)
    parser.add_argument("--epoch_number", help="Number of epochs to train.", default=20, type=int)
    parser.add_argument("--tau", help="Soft target update parameter.", default=0.01, type=float)
    parser.add_argument("--gamma", help="Discount factor.", default=1.00, type=float)
    parser.add_argument("--algorithm", type=str, default="DDPG", help="Algorithm type used. Valid options: DDPG, TD3 ")
    parser.add_argument("--replay_size", help="Size of the used replay memory.", default=1e6)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(os.getcwd(), f"data/{args.save_dir}")):
        os.makedirs(os.path.join(os.getcwd(), f"data/{args.save_dir}"), 0o777)
    if not os.path.exists(os.path.join(os.getcwd(), "tensorboard_runs")):
        os.mkdir("./tensorboard_runs")

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(f"./tensorboard_runs/{args.run_name}")
    num_epochs = args.epoch_number
    batch_size = args.batch_size
    tau = args.tau
    gamma = args.gamma
    replay_size = args.replay_size
    noise_stddev = 0.2
    if args.env == 1:
        print("ENV 1")
        env = LinEnv()
        optimal_k1 = -4.9611
        optimal_k2 = -4.4107
        a = np.array([[1.05, 0.05], [0.0, 1.05]])
        b = np.array([[0.0], [0.05]])
    else:
        print("ENV 2")
        env = LinEnv_2()
        a = np.array([[1.0, 0.1], [-0.05, 1.0]])
        b = np.array([[0.05], [0.0]])
        optimal_k1 = -1.2875
        optimal_k2 = 0.1637
    if args.algorithm.upper() == "TD3":
        agent = TD3(gamma,
                    tau,
                    env.observation_space.shape[0],
                    env.action_space
                    )
    elif args.algorithm.upper() == "DDPG":
        agent = DDPG(gamma=gamma, tau=tau, num_inputs=env.observation_space.shape[0], action_space=env.action_space)
    memory = ReplayMemory(replay_size)

    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(noise_stddev) * np.ones(nb_actions))

    rewards, policy_losses, value_losses, = [], [], []
    epoch = 0
    t = 0
    k_1_values = []
    k_2_values = []
    e_1_abs = []
    e_2_abs = []
    abs_eigenvalues = np.empty((0, 2), dtype=np.float32)
    while epoch <= num_epochs:
        ou_noise.reset()
        epoch_return = 0
        state = np.array([env.reset()])
        state = torch.Tensor(state).to(device)
        epoch_value_loss = 0
        epoch_policy_loss = 0
        counter = 0
        end_counter = 0

        while True:
            train_policy = False
            counter += 1
            if counter % 10 == 0:
                train_policy = True
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            epoch_return += reward
            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            memory.push(state, action, mask, next_state, reward)
            state = next_state
            numpy_state = state.cpu().detach().numpy()
            x_1 = numpy_state[0, 0]
            x_2 = numpy_state[0, 1]

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_params(batch, train_policy=train_policy)
                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
            if abs(x_1) > 10.0 or abs(x_2 > 10.0):
                break

            if done or counter > 200:
                break

        rewards.append(epoch_return / counter)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        print(f"Epoch: {epoch}  Loss {epoch_value_loss / counter}   Reward: {epoch_return} Epoch Steps {counter} ")
        # Get controller values
        agent.actor.eval()
        zero_input = np.array([[1.0, 0.0]]).astype(np.float32)
        zero_input = torch.from_numpy(zero_input).to(device)
        k_1 = agent.actor(zero_input)
        zero_input = np.array([[0.0, 1.0]]).astype(np.float32)
        zero_input = torch.from_numpy(zero_input).to(device)
        k_2 = agent.actor(zero_input)
        k_1, k_2 = k_1.cpu().detach().item(), k_2.cpu().detach().item()
        k_1_values.append(k_1)
        k_2_values.append(k_2)
        # Get absolute values of closed-loop eigenvalues
        w = calc_eigen(a=a, b=b, k1=k_1, k2=k_2)
        abs_1 = abs(w[0])
        abs_2 = abs(w[1])
        e_1_abs.append(abs_1)
        e_2_abs.append(abs_2)

        writer.add_scalar('epoch/value_loss', epoch_value_loss, epoch)
        writer.add_scalar('epoch/return', epoch_return, epoch)
        writer.add_scalar('epoch/k1', k_1, epoch)
        writer.add_scalar('epoch/k2', k_2, epoch)
        writer.add_scalar('epoch/abs_e1', abs_1, epoch)
        writer.add_scalar('epoch/abs_2', abs_2, epoch)

        agent.set_train()
        epoch += 1
    # Save Training results
    returns = np.array(rewards)
    np.save(f"./data/{args.save_dir}/returns.npy", returns)
    losses = np.array(value_losses)
    np.save(f"./data/{args.save_dir}/av_loss.npy", losses)
    K1 = np.array(k_1_values)
    K2 = np.array(k_2_values)
    Abs_E1 = np.array(e_1_abs)
    Abs_E2 = np.array(e_2_abs)
    np.save(f"./data/{args.save_dir}/k1.npy", K1)
    np.save(f"./data/{args.save_dir}/k2.npy", K2)
    np.save(f"./data/{args.save_dir}/abs_e1.npy", Abs_E1)
    np.save(f"./data/{args.save_dir}/abs_e2.npy", Abs_E2)
