import gym
from gym import spaces
import numpy as np


class LinEnv_2(gym.Env):

    def __init__(self, ):
        self.dt = 0.05
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf], dtype=np.float32),
                                            high=np.array([np.inf, np.inf], dtype=np.float32),
                                            dtype=np.float32)
        self.initial_weight = 2
        self.state = None

    def step(self, u):
        x_1, x_2 = self.state
        new_x1 = x_1 + 0.1 * x_2 + 0.05 * u
        new_x2 = -0.05 * x_1 + x_2
        costs = 1 * (x_1 ** 2) + 1 * (x_2 ** 2) + 1 * (u ** 2)
        state = [new_x1, new_x2]
        self.state = np.array(state, dtype=np.float32)
        if abs(x_1) < 1e-2 and abs(x_2) < 1e-2 and abs(new_x1 - x_1) < 1e-2 and abs(new_x2 - x_2) < 1e-2:
            done = True
        else:
            done = False
        return self._get_obs(), -costs, done, {}

    def _get_obs(self):
        x1, x2 = self.state
        return np.array([x1, x2], dtype=np.float32)

    def set_state(self, state):
        self.state = state

    def reset(self):
        high = np.array([self.initial_weight, self.initial_weight])
        self.state = np.random.uniform(low=-high, high=high)
        return self.state


class LinEnv(gym.Env):

    def __init__(self, ):
        self.dt = 0.05
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf], dtype=np.float32),
                                            high=np.array([np.inf, np.inf], dtype=np.float32),
                                            dtype=np.float32)
        self.initial_weight = 2.0
        self.state = None

    def step(self, u):
        x_1, x_2 = self.state
        new_x1 = 1.05 * x_1 + 0.05 * x_2
        new_x2 = 1.05 * x_2 + 0.05 * u
        costs = 1 * (1 * (x_1 ** 2) + 1 * (x_2 ** 2) + 1 * (u ** 2))
        state = [new_x1, new_x2]

        if abs(x_1) < 1e-2 and abs(x_2) < 1e-2 and abs(new_x1 - x_1) < 1e-2 and abs(new_x2 - x_2) < 1e-2:
            done = True
        else:
            done = False
        self.state = np.array(state, dtype=np.float32)
        return self._get_obs(), -costs, done, {}

    def _get_obs(self):
        x1, x2 = self.state

        return np.array([x1, x2], dtype=np.float32)

    def set_state(self, state):
        self.state = state

    def reset(self):
        high = np.array([self.initial_weight, self.initial_weight])
        self.state = np.random.uniform(low=-high, high=high)
        return self.state
