"""
Individual-level Vaccine Allocation Environment (VaxEnv)

A Gymnasium-compatible environment for vaccine allocation on contact networks
with CTMP-based epidemic dynamics.
"""

import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, Any


class VaxEnv(gym.Env):
    """
    Vaccine allocation environment with individual-level CTMP dynamics.

    State: Dictionary containing:
        - node_states: (N,) array of epidemic states (0=S, 1=E, 2=I, 3=R, 4=V, 5=D)
        - vaccinated: (N,) binary array indicating vaccination status
        - group_id: (N,) array indicating group membership (0, 1, 2)
        - time_step: Current time step
        - supply_today: Number of vaccine doses available today

    Action: (N,) array of vaccination probabilities for each individual [0, 1]
    """

    def __init__(
        self,
        N: int = 2000,
        G: int = 3,
        f_high_risk: float = 0.168,
        f_high_contact: float = 0.15,
        max_steps: int = 600,
        vax_supply_per_step: int = 10,
        r_0: float = 2.0,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.N = N
        self.G = G
        self.f_high_risk = f_high_risk
        self.f_high_contact = f_high_contact
        self.f_base = 1 - f_high_risk - f_high_contact
        self.max_steps = max_steps
        self.vax_supply_per_step = vax_supply_per_step
        self.r_0 = r_0
        self.seed = seed

        # Group sizes
        self.group_sizes = [
            int(self.f_base * N),
            int(self.f_high_risk * N),
            int(self.f_high_contact * N)
        ]
        # Adjust for rounding
        self.group_sizes[0] = N - sum(self.group_sizes[1:])

        # Disease parameters
        self.beta = 0.05  # Base transmission rate (will be scaled by R0)
        self.sigma = 1.0 / 4.0  # E -> I rate (4 days incubation)
        self.gamma = 1.0 / 7.0  # I -> R rate (7 days infectious)
        self.vax_efficacy = 0.8

        # Death rates by group
        self.death_rates = np.array([0.01, 0.1, 0.01])  # [base, high_risk, high_contact]

        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'node_states': gym.spaces.Box(low=0, high=5, shape=(N,), dtype=np.int32),
            'vaccinated': gym.spaces.Box(low=0, high=1, shape=(N,), dtype=np.int32),
            'group_id': gym.spaces.Box(low=0, high=G-1, shape=(N,), dtype=np.int32),
            'time_step': gym.spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
            'supply_today': gym.spaces.Box(low=0, high=vax_supply_per_step*2, shape=(1,), dtype=np.float32),
        })

        # Initialize graph (will be created in reset)
        self.graph = None
        self.group_id = None
        self.node_states = None
        self.vaccinated = None
        self.time_step = 0

    def _create_contact_graph(self) -> nx.Graph:
        """
        Create a contact network with group structure.
        Uses configuration model with degree distributions.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        # Assign groups
        group_id = np.zeros(self.N, dtype=int)
        idx = 0
        for g in range(self.G):
            group_id[idx:idx+self.group_sizes[g]] = g
            idx += self.group_sizes[g]

        # Add edges based on contact matrix
        # Contact matrix (expected edges between groups)
        contact_matrix = np.array([
            [0.165, 0.1, 0.175],
            [0.1, 0.0, 0.002],
            [0.175, 0.002, 0.132]
        ])

        # Scale to get average degree
        avg_degree = 10
        contact_matrix = contact_matrix * avg_degree * self.N / 2

        # Create edges
        for i in range(self.N):
            for j in range(i+1, self.N):
                g_i = group_id[i]
                g_j = group_id[j]

                # Edge probability based on contact matrix
                p_edge = contact_matrix[g_i, g_j] / (self.group_sizes[g_i] * self.group_sizes[g_j])
                p_edge = min(p_edge, 1.0)

                if np.random.rand() < p_edge:
                    G.add_edge(i, j, contact_prob=0.1)

        # Store group assignments
        for i in range(self.N):
            G.nodes[i]['group'] = group_id[i]
            G.nodes[i]['susceptibility'] = 1.0
            G.nodes[i]['infectiousness'] = 1.0
            # High risk group has higher susceptibility
            if group_id[i] == 1:
                G.nodes[i]['susceptibility'] = 1.5

        return G, group_id

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Create contact graph
        self.graph, self.group_id = self._create_contact_graph()

        # Initialize epidemic states: 0=S, 1=E, 2=I, 3=R, 4=V, 5=D
        self.node_states = np.zeros(self.N, dtype=np.int32)  # All susceptible

        # Randomly infect a few individuals
        n_initial_infected = 15
        initial_infected = np.random.choice(self.N, size=n_initial_infected, replace=False)
        self.node_states[initial_infected] = 1  # Exposed

        # Initialize vaccination status
        self.vaccinated = np.zeros(self.N, dtype=np.int32)

        self.time_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> Dict:
        """Get current observation."""
        return {
            'node_states': self.node_states.copy(),
            'vaccinated': self.vaccinated.copy(),
            'group_id': self.group_id.copy(),
            'time_step': np.array([self.time_step], dtype=np.int32),
            'supply_today': np.array([self.vax_supply_per_step], dtype=np.float32),
        }

    def _get_info(self) -> Dict:
        """Get additional info."""
        return {
            'n_susceptible': np.sum(self.node_states == 0),
            'n_exposed': np.sum(self.node_states == 1),
            'n_infected': np.sum(self.node_states == 2),
            'n_recovered': np.sum(self.node_states == 3),
            'n_vaccinated': np.sum(self.node_states == 4),
            'n_dead': np.sum(self.node_states == 5),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: (N,) array of vaccination probabilities

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Vaccination phase
        action = np.clip(action, 0, 1)

        # Only vaccinate susceptible individuals
        eligible = (self.node_states == 0) & (self.vaccinated == 0)
        action = action * eligible

        # Normalize to respect supply constraint
        total_action = action.sum()
        if total_action > self.vax_supply_per_step:
            action = action * (self.vax_supply_per_step / (total_action + 1e-8))

        # Sample actual vaccinations (stochastic)
        vaccinated_today = np.random.binomial(1, action)
        self.vaccinated[vaccinated_today == 1] = 1
        self.node_states[vaccinated_today == 1] = 4  # Move to vaccinated state

        # 2. Disease progression (CTMP step)
        new_states = self.node_states.copy()

        # S -> E (infection)
        susceptible = np.where(self.node_states == 0)[0]
        for node in susceptible:
            # Count infected neighbors
            neighbors = list(self.graph.neighbors(node))
            infected_neighbors = sum(1 for n in neighbors if self.node_states[n] == 2)

            # Force of infection
            foi = self.beta * infected_neighbors / max(len(neighbors), 1)

            if np.random.rand() < foi:
                new_states[node] = 1  # Exposed

        # E -> I
        exposed = np.where(self.node_states == 1)[0]
        for node in exposed:
            if np.random.rand() < self.sigma:
                new_states[node] = 2  # Infected

        # I -> R or D
        infected = np.where(self.node_states == 2)[0]
        for node in infected:
            if np.random.rand() < self.gamma:
                # Check if dies
                group = self.group_id[node]
                if np.random.rand() < self.death_rates[group]:
                    new_states[node] = 5  # Dead
                else:
                    new_states[node] = 3  # Recovered

        # V breakthrough infections (reduced by efficacy)
        vaccinated_susceptible = np.where(self.node_states == 4)[0]
        for node in vaccinated_susceptible:
            neighbors = list(self.graph.neighbors(node))
            infected_neighbors = sum(1 for n in neighbors if self.node_states[n] == 2)
            foi = self.beta * infected_neighbors / max(len(neighbors), 1) * (1 - self.vax_efficacy)

            if np.random.rand() < foi:
                new_states[node] = 1  # Breakthrough to exposed

        self.node_states = new_states

        # 3. Calculate reward (negative infections and deaths)
        n_infected = np.sum(self.node_states == 2)
        n_dead = np.sum(self.node_states == 5)
        reward = -(n_infected + 10 * n_dead)  # Penalize deaths more

        # 4. Check termination
        self.time_step += 1
        terminated = self.time_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Optional rendering."""
        info = self._get_info()
        print(f"Step {self.time_step}: S={info['n_susceptible']}, E={info['n_exposed']}, "
              f"I={info['n_infected']}, R={info['n_recovered']}, V={info['n_vaccinated']}, D={info['n_dead']}")
