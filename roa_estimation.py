import numpy as np
import torch
from average import Controller
from matplotlib import pyplot as plt
import matplotlib
import argparse
from matplotlib import cm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nonlinear_actor = Controller(state_dim=2, action_dim=1, max_action=2.0).to(device)
nonlinear_actor.load_state_dict(torch.load("./final_controller/nonlinear_controller.pt"))

class Data(torch.utils.data.Dataset):
    def __init__(self, data):
        super(Data, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, item):
        return self.data[item]

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Environment_Sim():
    def __init__(self, length, mass, g=9.81, max_torque=2.0, max_speed=8.0):
        self.l = length
        self.m = mass
        self.g = g
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.dt = 0.05
    def next_states(self, states, actions):
        states = states.detach().numpy()
        actions = actions.detach().numpy()
        actions = actions.clip(-self.max_torque, self.max_torque)
        th = states[:, 0]
        thdot = states[:, 1]
        costs = np.power(th, 2) + np.power(thdot, 2) + np.power(actions, 2)
        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * actions) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + thdot * self.dt
        newth = angle_normalize(newth)
        newth, newthdot = np.expand_dims(newth,1), np.expand_dims(newthdot,1)
        next_states = np.concatenate((newth, newthdot), axis=1)
        next_states = torch.from_numpy(next_states)
        costs = torch.from_numpy(costs)
        return next_states, costs

    def calc_trajectories(self, start_states, actor, steps=500):
        st = start_states
        cst = torch.zeros((st.shape[0],))
        for k in range(steps):
            samples = Data(data=st)
            dataloader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=False)
            if (k+1) % 100 == 0:
                print("Trajectory Step", k+1, " ...")
            # print(st)
            #torch_state = torch.from_numpy(st).to(device)
            actions = torch.empty(size=(0,))
            next_states = torch.empty(size=(0,2))
            c = torch.empty(size=(0,))
            for batch in dataloader:
                dat = batch.to(device)
                batch_actions = actor(dat).detach().to("cpu")[:,0]
                batch_next_states, batch_costs = self.next_states(batch, batch_actions)
                next_states = torch.cat((next_states, batch_next_states))
                c = torch.cat((c, batch_costs))
                actions = torch.cat((actions, batch_actions))
            cst += c
            st = next_states

        return st, cst



def linear_controller(states):
    # actions = states[:,0] * -9.0176 + states[:,1] * -2.4519
    actions = (-9.0176 * states[:, 0] - 2.4519 * states[:, 1]).unsqueeze(1)
    return actions


def lqr_controller(states):
    actions = (-9.6872 * states[:, 0] - 0.4629 * states[:, 1]).unsqueeze(1)
    return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ROA Estimation of x=  0, for closed-loop '
                                                 'pendulum system')
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--length", type=float, default=1.0, help="Length of pendulum.")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass of pendulum.")
    parser.add_argument("--max_action", type=float, default=2.0, help="Mass of pendulum.")
    args = parser.parse_args()

    start_theta = np.arange(start=-np.pi, stop=np.pi, step=0.04).astype(np.float32)
    start_thdot = np.arange(start=-8, stop=8, step=0.04).astype(np.float32)
    a, b = np.meshgrid(start_theta, start_thdot)
    xa = np.reshape(a, newshape=(-1, 1))
    xb = np.reshape(b, newshape=(-1, 1))
    states = np.concatenate((xa, xb), axis=1).astype(np.float32)
    #states = np.array([[0.22940193, 1.05789482]]).astype(np.float32)
    del xa, xb
    states = torch.from_numpy(states)
    actor = Controller(state_dim=2, action_dim=1, max_action=args.max_action).to(device)
    actor.load_state_dict(torch.load(args.model_dir + "/nonlinear_controller.pt"))
    #actor = linear_controller
    sim = Environment_Sim(mass=args.mass, length=args.length, max_torque=args.max_action)
    st, cst = sim.calc_trajectories(start_states=states, actor=actor, steps=500)
    st = torch.pow(st, 2)
    st = torch.sum(st, 1)
    st = torch.sqrt(st)
    test = torch.where(st < 1e-5, 1, 0)
    print(torch.sum(test).item())
    print(torch.mean(cst))

    # st = st.detach().numpy()
    st = st.detach().cpu().numpy()
    cst = cst.detach().cpu().numpy()
    np.save(args.model_dir + "/trajectory_error.npy", st)
    np.save(args.model_dir + "/trajectory_costs.npy", cst)