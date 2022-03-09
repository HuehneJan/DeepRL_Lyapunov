from average import Controller
import torch
import argparse
import warnings
import numpy as np
warnings.simplefilter("ignore", UserWarning)


def _get_lin_controller_values(model):
    inp = torch.zeros((1, 2))
    inp.requires_grad = True
    prediction = model(inp)
    prediction.backward(retain_graph=True)
    values = inp.grad.detach().numpy()[0,:]
    return values


def _calc_eigen(k1, k2,length=1, mass=1):
    e_11 = 1
    e_12 = 0.05
    e_21 = (3*9.81*0.05)/(2*length) + k1 * (3*0.05) / (mass*length**2)
    e_22 = 1 + k2 * (3*0.05) / (mass*length**2)
    A = np.array([[e_11, e_12], [e_21, e_22]])
    print("System Matrix A:", A)
    e, _ = np.linalg.eig(A)
    print(f"The absolute of the eigenvalues of the linearized system are {abs(e[0])} and {abs(e[1])}")
    if abs(e[0]) < 1 and abs(e[1]) < 1:
        print("The nonlinear system is asymptotically stable near the equilibrium.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linearize closed-loop pendulum system with average nonlinear "
                                                 "controller.")
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--max_action", type=float, default=2.0)
    args = parser.parse_args()
    actor = Controller(state_dim=2, action_dim=1, max_action=args.max_action)
    actor.load_state_dict(torch.load(args.model_dir +"/nonlinear_controller.pt"))
    actor = actor.to("cpu")
    val = _get_lin_controller_values(actor)
    print("Linearized Controller Values ", val)
    _calc_eigen(k1=val[0], k2=val[1], mass=args.mass, length=args.length)
