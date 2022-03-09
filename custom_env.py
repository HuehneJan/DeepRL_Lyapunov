import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math




class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, m=1.0, l=1.0, noise=0, max_action=2.0):
        self.max_speed = 8
        self.max_torque = max_action
        self.dt = 0.05
        self.g = 9.81
        self.m = m
        self.noise = noise
        self.l = l
        self.viewer = None
        self.state = None
        #high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        high = np.array([math.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        #print(u.dtype)
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 +  thdot ** 2 + (u ** 2)
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + thdot * dt
        newth = angle_normalize(newth)

        if abs(th) < 1e-3 and abs(thdot) < 1e-3 and abs(newth) < 1e-3 and abs(newthdot) < 1e-3:
            done = True
        else:
            done = False
        if self.noise != 0:
            newth = newth + np.random.normal(0, self.noise, 1)[0]
            newthdot = newthdot + np.random.normal(0, self.noise, 1)[0]

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([math.pi, 1.0])
        self.state =  np.random.uniform(low=-high, high=high)
        self.last_u = None
        #self.state = np.array([0.3, 0.3])
        return self._get_obs()

    def _get_obs(self):
        return self.state
        #theta, thdot = self.state
        #sintheta = math.sin(theta)
        #costheta = math.cos(theta)
        #return np.array([sintheta, costheta, thdot]).astype(np.float32)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi