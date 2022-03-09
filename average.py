import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from custom_env import PendulumEnv
import argparse
import warnings

warnings.simplefilter("ignore", UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Dataset Class without labels
class Data(torch.utils.data.Dataset):
    def __init__(self, data):
        super(Data, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, item):
        return self.data[item]


# Dataset Class with labels
class DataSet(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Average Controller Class
class Controller(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Controller, self).__init__()
        self.l1 = nn.Linear(state_dim, 16, bias=False)
        self.l2 = nn.Linear(16, 6, bias=False)
        self.l3 = nn.Linear(6, action_dim, bias=False)
        self.max_action = max_action

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

def get_average_lin_values(model_dir,
                           start_epoch=20000,
                           end_epoch=40000):
    epochs = np.arange(start=start_epoch, stop=end_epoch, step=100)
    k_1 = 0.0
    k_2 = 0.0
    for epoch in epochs:
        actor = torch.load(model_dir + f"/actor{epoch}.pt").to(device)
        one_input = torch.FloatTensor([[0.0, 1.0]]).to(device)
        k_1_epoch = actor(one_input).detach().cpu().item()
        one_input = torch.FloatTensor([[1.0, 0.0]]).to(device)
        k_2_epoch = actor(one_input).detach().cpu().item()

        k_1 += k_1_epoch
        k_2 += k_2_epoch
    k_1 = k_1 / len(epochs)
    k_2 = k_2 / len(epochs)
    print("The average linear controller values are  ", k_1, "  and  ", k_2)
def create_samples(x_samples,
                   model_dir,
                   batch_size=256,
                   start_epoch=5000,
                   end_epoch=40000,
                   ):

    """
    Method to get control samples from saved model in interval [start_epoch, end_epoch].
    Will save mean controller values as numpy file into model_dir.
    :param x_samples: X samples passed to every model.
    :param model_dir: Path to directory where models are saved.
    :param batch_size: Batch size used for model evaluation.
    :param start_epoch: Start value of model interval
    :param end_epoch: End Value of model interval
    :return: None
    """
    batch_size = batch_size
    epochs = np.arange(start=start_epoch, stop=end_epoch, step=100)
    data = x_samples
    dataset = Data(torch.from_numpy(data))
    samples = torch.zeros(size=(len(dataset), 1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in epochs:
        print(epoch)
        actor = torch.load(model_dir + f"/actor{epoch}.pt").to(device)

        for i_batch, batch in enumerate(dataloader):
            batch = batch.to(device)
            actions = actor(batch).cpu().detach()
            samples[batch_size * i_batch:(batch_size * i_batch) + batch_size, :] += actions

    samples = samples.detach().numpy()
    samples = samples / len(epochs)
    np.save(model_dir + "/samples_data.npy", samples)


def create_dataset(data_dir):
    """
    Method to create the torch dataset. Will call create_samples to create control samples if needed.
    :param data_dir: Path to directory where data should be saved / is saved. If control samples need to created, the
    directory must contain the model data.
    :return: torch.Dataset instance
    """
    theta = np.arange(start=-math.pi, stop=math.pi + 0.001, step=0.01, dtype=np.float32)
    theta_dot = np.arange(start=-8.0, stop=8.0, step=0.01, dtype=np.float32)
    xa, xb = np.meshgrid(theta, theta_dot)
    xa = np.reshape(xa, newshape=(-1, 1))
    xb = np.reshape(xb, newshape=(-1, 1))
    data = np.concatenate((xa, xb), axis=1)
    if not os.path.exists(data_dir + "/samples_data.npy"):
        print("DOES NOT EXIST")
        create_samples(model_dir=data_dir, x_samples=data)
    targets = np.load(data_dir + "/samples_data.npy")
    del xa, xb, theta, theta_dot
    d_set = DataSet(data=data, labels=targets)
    return d_set


def train(it,
          epochs,
          save_dir,
          length=1,
          max_action=2.0,
          mass=1,
          reward_runs_per_epoch=10):
    model = Controller(state_dim=2, action_dim=1, max_action=max_action).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    env = PendulumEnv(l=length, m=mass, max_action=max_action)
    epoch_loss_ = np.empty(shape=(0,))
    epoch_average_return_ = np.empty(shape=(0,))
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, target in it:
            batch = batch.to(device)
            target = target.to(device)
            values = model(batch)
            loss = F.mse_loss(values, target)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_reward = 0.0
        for i in range(reward_runs_per_epoch):
            run_reward = 0.0
            env.reset()
            state = env.state
            vis_state = np.array([state[0], state[1]], dtype=np.float32)
            for k in range(200):
                torch_state = torch.from_numpy(vis_state).unsqueeze(0).to(device)
                action = model(torch_state)
                action = action.cpu().detach().numpy()[0]
                vis_state, reward, done, _ = env.step(action)
                run_reward += reward
                vis_state = vis_state.astype(np.float32)
                if done:
                    break
            epoch_reward += run_reward
        epoch_loss_ = np.append(epoch_loss_, epoch_loss.detach().item() / len(it))
        epoch_average_return_ = np.append(epoch_average_return_, epoch_reward / reward_runs_per_epoch)
        print(f"Epoch {epoch}     Loss  {epoch_loss / len(it)}     "
              f"Epoch Reward  {epoch_reward / reward_runs_per_epoch}")
    np.save(save_dir + "/nonlinear_loss.npy", epoch_loss_)
    np.save(save_dir + "/nonlinear_reward.npy", epoch_average_return_)
    torch.save(model.state_dict(), save_dir + "/nonlinear_controller.pt")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average Nonlinear Controller Training for Pendulum System.")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--length", type=float, default=1)
    parser.add_argument("--mass", type=float, default=1)
    parser.add_argument("--max_action", type=float, default=2.0)
    parser.add_argument("--lin_actor", action="store_true", default=False)
    args = parser.parse_args()
    if args.lin_actor:
        get_average_lin_values(model_dir=args.model_dir)
    else:
        train(it=torch.utils.data.DataLoader(create_dataset(data_dir=args.model_dir),
                                             batch_size=256,
                                             shuffle=True),
              epochs=50,
              save_dir=args.save_dir,
              length=args.length,
              mass=args.mass,
              max_action=args.max_action
              )
