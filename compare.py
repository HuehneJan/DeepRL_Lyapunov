from roa_estimation import *
import torch
import numpy as np
from average import Controller
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    sim = Environment_Sim(mass=1, length=1, max_torque=2.0)
    start_theta = np.arange(start=-np.pi, stop=np.pi, step=0.04).astype(np.float32)
    start_thdot = np.arange(start=-8, stop=8, step=0.04).astype(np.float32)
    a, b = np.meshgrid(start_theta, start_thdot)
    xa = np.reshape(a, newshape=(-1, 1))
    xb = np.reshape(b, newshape=(-1, 1))
    states = np.concatenate((xa, xb), axis=1).astype(np.float32)
    del xa, xb
    states = torch.from_numpy(states)
    print("LQR Controller")
    actor = lqr_controller
    lqr_st, lqr_cst = sim.calc_trajectories(start_states=states, actor=actor, steps=500)
    st = torch.pow(lqr_st, 2)
    st = torch.sum(st, 1)
    st = torch.sqrt(st)
    test = torch.where(st < 1e-3, 1, 0)
    print(f"Number Converged {torch.sum(test).item()} from a total of {lqr_st.size()[0]} states.")

    print("Average Linear Controller")
    actor = linear_controller
    lin_st, lin_cst = sim.calc_trajectories(start_states=states, actor=actor, steps=500)
    st = torch.pow(lin_st, 2)
    st = torch.sum(st, 1)
    st = torch.sqrt(st)
    test = torch.where(st < 1e-3, 1, 0)
    print(f"Number Converged {torch.sum(test).item()} from a total of {lin_st.size()[0]} states.")

    print("RSN-DNN Controller")
    actor = Controller(state_dim=2, action_dim=1, max_action=2.0)
    actor.load_state_dict(torch.load("./nonlinear/nonlinear_controller.pt"))
    actor = actor.to(device)
    nonlin_st, nonlin_cst = sim.calc_trajectories(start_states=states, actor=actor, steps=500)
    st = torch.pow(nonlin_st, 2)
    st = torch.sum(st, 1)
    st = torch.sqrt(st)
    test = torch.where(st < 1e-3, 1, 0)
    print(f"Number Converged {torch.sum(test).item()} from a total of {nonlin_st.size()[0]} states.")

    print("Nonlinear-Linear Comparison")
    better = torch.where(-nonlin_cst+lin_cst >= 0.0, 1, 0)
    print(f"Better Performance in {torch.sum(better).item()} from a total of {nonlin_cst.size()[0]} states.")