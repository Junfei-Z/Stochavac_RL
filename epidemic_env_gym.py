# epidemic_env_gym.py

import gym
from gym import spaces
import numpy as np
from epidemic_env import EpidemicEnv
from graph_model import create_contact_graph

class EpidemicGymWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.G = create_contact_graph()
        self.env = EpidemicEnv(self.G)
        self.num_nodes = len(self.G.nodes)
        
        # Action space: binary decision to vaccinate or not for each node
        self.action_space = spaces.MultiBinary(self.num_nodes)
        
        # Observation: encode each node's state as integer (S=0, E=1, ..., D=8)
        self.observation_space = spaces.MultiDiscrete([9] * self.num_nodes)

    def reset(self):
        state = self.env.reset()
        return self._encode(state)

    def step(self, action):
        action_dict = {i: action[i] for i in range(self.num_nodes)}
        state, reward, done, _ = self.env.step(action_dict)
        return self._encode(state), reward, done, {}

    def _encode(self, state_dict):
        mapping = {'S': 0, 'E': 1, 'P': 2, 'A': 3, 'I': 4, 'L': 5, 'H': 6, 'R': 7, 'V': 8, 'D': 9}
        return np.array([mapping[state_dict[i]] for i in range(self.num_nodes)])
