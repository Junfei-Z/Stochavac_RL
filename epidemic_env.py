# epidemic_env.py

import numpy as np
from config import EPISODE_LENGTH, VACCINE_BUDGET

class EpidemicEnv:
    def __init__(self, G):
        self.G = G
        self.time = 0
        self.state = {}
        self.max_time = EPISODE_LENGTH

    def reset(self):
        self.time = 0
        self.state = {i: 'S' for i in self.G.nodes}
        initial = np.random.choice(list(self.G.nodes))
        self.state[initial] = 'I'
        return self.state

    def step(self, actions):
        new_state = self.state.copy()
        for node in self.G.nodes:
            s = self.state[node]
            p = self.G.nodes[node]

            # Vaccinate
            if s == 'S' and actions.get(node, 0) == 1:
                new_state[node] = 'V'
                continue

            # S → E
            if s == 'S':
                risk = 0
                for neighbor in self.G.neighbors(node):
                    if self.state[neighbor] in ['I']:
                        q = self.G.nodes[neighbor]
                        c = self.G.edges[node, neighbor]['contact_prob']
                        risk += c * p['susceptibility'] * q['infectiousness']
                if np.random.rand() < risk:
                    new_state[node] = 'E'

            # E → I
            elif s == 'E':
                if np.random.rand() < p['progression_rate']:
                    new_state[node] = 'I'

            # I → R
            elif s == 'I':
                if np.random.rand() < p['recovery_rate']:
                    new_state[node] = 'R'

        self.state = new_state
        self.time += 1
        done = self.time >= self.max_time
        reward = -sum(1 for s in self.state.values() if s == 'I')
        return self.state, reward, done, {}
