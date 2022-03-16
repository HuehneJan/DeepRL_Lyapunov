import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calc_eigen(a, b, k1, k2):
    b_k = np.array([[b[0,0]*k1, b[0,0]*k2], [b[1,0]*k1, b[1,0]*k2]])
    new_a = a + b_k
    w,v = np.linalg.eig(new_a)
    return w

def print_q_function(critic, state, action_low, action_high, k_1, k_2, optimal_k1=-4.9611, optimal_k2=-4.4107,
                     save_dir=None, name=None):
    critic.eval()
    action_values = np.arange(action_low, action_high, 0.1).astype(np.float32)
    actions = action_values.copy()
    action_values = np.expand_dims(action_values, axis=1)

    action_values = torch.from_numpy(action_values).to(device)

    q_values = []
    for i in range(action_values.size()[0]):
        action = torch.unsqueeze(action_values[i], axis=0)

        value = critic.Q1(state, action)
        q_values.append(value.cpu().detach().item())
    new_state = state.cpu().detach().numpy()
    reference_action = new_state[0,0]*k_1 + new_state[0,1]*k_2
    reference_action = np.array([[reference_action]]).astype(np.float32)
    reference_action = torch.from_numpy(reference_action).to(device)

    optimal_action = new_state[0,0]*optimal_k1 + new_state[0,1]* optimal_k2
    optimal_action = np.array([[optimal_action]]).astype(np.float32)
    optimal_action = torch.from_numpy(optimal_action).to(device)
    reference_q = critic.Q1(state, reference_action)
    reference_q = reference_q.cpu().detach().item()
    optimal_q = critic.Q1(state, optimal_action)
    optimal_q = optimal_q.cpu().detach().item()
    reference_action = reference_action.cpu().detach().item()
    optimal_action = optimal_action.cpu().detach().item()
    critic.train()

    plt.plot(actions, q_values, color="green")
    plt.plot(reference_action, reference_q, marker="o", color="red", label="real")
    plt.plot(optimal_action, optimal_q, marker="o", color="blue", label="desired")
    plt.grid()
    plt.legend()
    if save_dir != None and name != None:
        plt.savefig(save_dir + "/" + name, format="eps")
    plt.show()