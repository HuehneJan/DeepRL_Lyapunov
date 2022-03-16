import torch
import numpy as np
import torch.nn as nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.linear = nn.Linear(num_inputs, num_outputs, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)



class TD3_Critic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(TD3_Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Q1 architecture
        # Layer 1
        self.layer1_1 = nn.Linear(num_inputs + num_outputs, 40)
        self.norm1_1 = nn.LayerNorm(40)
        # Layer2
        self.layer1_2 = nn.Linear(40, 80)
        self.norm1_2 = nn.LayerNorm(80)
        # Layer 3
        self.layer1_3 = nn.Linear(80, 60)
        self.norm1_3 = nn.LayerNorm(60)
        # Layer 4
        self.layer1_4 = nn.Linear(60, 40)
        self.norm1_4 = nn.LayerNorm(40)
        # Layer 5
        self.layer1_5 = nn.Linear(40, 1)
        # Weight Init
        fan_in_uniform_init(self.layer1_1.weight)
        fan_in_uniform_init(self.layer1_1.bias)

        fan_in_uniform_init(self.layer1_2.weight)
        fan_in_uniform_init(self.layer1_2.bias)

        fan_in_uniform_init(self.layer1_3.weight)
        fan_in_uniform_init(self.layer1_3.bias)

        fan_in_uniform_init(self.layer1_4.weight)
        fan_in_uniform_init(self.layer1_4.bias)

        nn.init.uniform_(self.layer1_5.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.layer1_5.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        # Q2 architecture

        # Q1 architecture
        # Layer 1
        self.layer2_1 = nn.Linear(num_inputs + num_outputs, 40)
        self.norm2_1 = nn.LayerNorm(40)
        # Layer2
        self.layer2_2 = nn.Linear(40, 80)
        self.norm2_2 = nn.LayerNorm(80)
        # Layer 3
        self.layer2_3 = nn.Linear(80, 60)
        self.norm2_3 = nn.LayerNorm(60)
        # Layer 4
        self.layer2_4 = nn.Linear(60, 40)
        self.norm2_4 = nn.LayerNorm(40)
        # Layer 5
        self.layer2_5 = nn.Linear(40, 1)
        # Weight Init
        fan_in_uniform_init(self.layer2_1.weight)
        fan_in_uniform_init(self.layer2_1.bias)

        fan_in_uniform_init(self.layer2_2.weight)
        fan_in_uniform_init(self.layer2_2.bias)

        fan_in_uniform_init(self.layer2_3.weight)
        fan_in_uniform_init(self.layer2_3.bias)

        fan_in_uniform_init(self.layer2_4.weight)
        fan_in_uniform_init(self.layer2_4.bias)

        nn.init.uniform_(self.layer2_5.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.layer2_5.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        inp = torch.concat((inputs, actions), axis=1)
        q1 = self.layer1_1(inp)
        q1 = self.norm1_1(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_2(q1)
        q1 = self.norm1_2(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_3(q1)
        q1 = self.norm1_3(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_4(q1)
        q1 = self.norm1_4(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_5(q1)

        q2 = self.layer2_1(inp)
        q2 = self.norm2_1(q2)
        q2 = torch.relu(q2)

        q2 = self.layer2_2(q2)
        q2 = self.norm2_2(q2)
        q2 = torch.relu(q2)

        q2 = self.layer2_3(q2)
        q2 = self.norm2_3(q2)
        q2 = torch.relu(q2)

        q2 = self.layer2_4(q2)
        q2 = self.norm2_4(q2)
        q2 = torch.relu(q2)

        q2 = self.layer2_5(q2)
        return q1, q2

    def Q1(self, state, action):
        inp = torch.concat((state, action), axis=1)
        q1 = self.layer1_1(inp)
        q1 = self.norm1_1(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_2(q1)
        q1 = self.norm1_2(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_3(q1)
        q1 = self.norm1_3(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_4(q1)
        q1 = self.norm1_4(q1)
        q1 = torch.relu(q1)

        q1 = self.layer1_5(q1)
        return q1



class Critic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Q1 architecture
        # Layer 1
        self.layer1 = nn.Linear(num_inputs + num_outputs, 40)
        self.norm1 = nn.LayerNorm(40)
        # Layer2
        self.layer2 = nn.Linear(40, 80)
        self.norm2 = nn.LayerNorm(80)
        # Layer 3
        self.layer3 = nn.Linear(80, 60)
        self.norm3 = nn.LayerNorm(60)
        # Layer 4
        self.layer4 = nn.Linear(60, 40)
        self.norm4 = nn.LayerNorm(40)
        # Layer 5
        self.layer5 = nn.Linear(40, 1)
        # Weight Init
        fan_in_uniform_init(self.layer1.weight)
        fan_in_uniform_init(self.layer1.bias)

        fan_in_uniform_init(self.layer2.weight)
        fan_in_uniform_init(self.layer2.bias)

        fan_in_uniform_init(self.layer3.weight)
        fan_in_uniform_init(self.layer3.bias)

        fan_in_uniform_init(self.layer4.weight)
        fan_in_uniform_init(self.layer4.bias)

        nn.init.uniform_(self.layer5.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.layer5.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = torch.concat((inputs, actions), axis = 1)
        x = self.layer1(x)
        x = self.norm1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = self.norm2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = self.norm3(x)
        x = torch.relu(x)

        x = self.layer4(x)
        x = self.norm4(x)
        x = torch.relu(x)

        x = self.layer5(x)
        return x
