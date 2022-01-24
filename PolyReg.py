import gym
import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
theta = np.arange(-np.pi, np.pi+0.01, 0.01)
thetadot = np.arange(-8, 8, 0.01)
xa, xb = np.meshgrid(theta, thetadot)
xa = np.reshape(xa, newshape=(-1,1))
xb = np.reshape(xb, newshape=(-1, 1))

X = np.concatenate((xa, xb), axis=1)

y = np.load("./test2/array/average_actions.npy")[:,0]


poly_reg = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly_reg.fit_transform(X)

lin_reg =LinearRegression(fit_intercept=False)
lin_reg.fit(X_poly, y)

env = gym.make("Pendulum-v1")
env.reset()
state = env.state
states = np.empty(shape=(0,2), dtype=np.float32)
sys_state = np.reshape(env.state, newshape=(1, -1))
states = np.append(states, sys_state, axis=0)
time_steps = [0.0]
for k in range(200):
    time_steps.append((k + 1) * 0.05)
    poly_state = poly_reg.fit_transform(np.reshape(state, newshape=(1, -1)))
    action = lin_reg.predict(poly_state)
    next_state, costs, done, _ = env.step(action)
    env.render()
    state = env.state
    sys_state = np.reshape(env.state, newshape=(1, -1))
    states = np.append(states, sys_state, axis=0)
    if done:
        break
env.close()
print(state)
plt.plot(time_steps, states[:,0])
plt.grid()
plt.show()
plt.plot(time_steps, states[:,1])
plt.grid()
plt.show()