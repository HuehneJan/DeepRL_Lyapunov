
import numpy as np
import torch
import math
import gym
from matplotlib import pyplot as plt


env = gym.make("Pendulum-v1")

env.reset()

states = np.empty((0,2))
states = np.append(states, [env.state], axis=0)
time_steps = [0.0]
vis_state = np.array(env.state).astype(np.float32)
for k in range(500):
    time_steps.append((k+1)*0.05)
    action = [-9.0176*vis_state[0] -2.4519*vis_state[1]]
    env.render()
    vis_state, reward, done, _ = env.step(action)
    sys_state = np.reshape(env.state, newshape=(1, -1))
    states = np.append(states, sys_state, axis=0)

env.close()
print(env.state, k)

plt.plot(time_steps, states[:,0])
plt.grid()
plt.show()
plt.plot(time_steps, states[:,1])
plt.grid()
plt.show()

"""
state = (2.0, 0.0, 0.4, 0.0)
env = CartPoleEnv()
vis_state = env.reset().astype(np.float32)
env.state = state
vis_state = np.array(state).astype(np.float32)
for k in range(1000):
    action = [vis_state[0]*0.9 + vis_state[1]*2 + vis_state[2]*30.0 + vis_state[3]*8]
    env.render()
    vis_state, reward, done, _ = env.step(action)
    x = vis_state[0]
    theta = vis_state[2]
   
    if abs(x) > 5.0:
        break
    if done:
        print(k)
        break
env.close()
"""