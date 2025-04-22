# graph_model.py

import networkx as nx
import numpy as np
from config import GROUPS, GROUP_SIZE, SEED

def create_contact_graph(seed=SEED):
    np.random.seed(seed)
    G = nx.Graph()
    idx = 0
    for group in GROUPS:
        for _ in range(GROUP_SIZE):
            G.add_node(idx,
                       group=group,
                       susceptibility=np.clip(np.random.normal(0.5, 0.1), 0.1, 1.0),
                       progression_rate=np.random.uniform(0.3, 1.0),
                       infectiousness=np.random.uniform(0.3, 1.0),
                       recovery_rate=np.random.uniform(0.3, 0.8))
            idx += 1

    for i in G.nodes:
        for j in G.nodes:
            if i < j and np.random.rand() < 0.6:
                G.add_edge(i, j, contact_prob=np.random.uniform(0.2, 0.8))
    return G
