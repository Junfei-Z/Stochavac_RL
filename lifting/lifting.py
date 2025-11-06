"""
Macro-to-Micro Lifting Module

This module lifts group-level vaccine allocation decisions (from ODE expert)
to individual-level allocations for the CTMP environment.
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional


def lift_macro_to_micro(
    Ug: np.ndarray,
    state: Dict,
    graph: nx.Graph,
    rule: str = 'degree_risk',
    eps: float = 1e-8
) -> np.ndarray:
    """
    Lift macro-level group allocation to micro-level individual allocation.

    Args:
        Ug: (G,) array of vaccine doses allocated to each group
        state: Dictionary containing:
            - N: Number of individuals
            - group_id: (N,) array of group IDs
            - vaccinated: (N,) binary array of vaccination status
            - supply_today: Total vaccine supply
            - risk: Optional (N,) array of individual risk scores
        graph: NetworkX graph with contact network
        rule: Allocation rule within groups ('degree', 'risk', 'degree_risk', 'uniform')
        eps: Small constant for numerical stability

    Returns:
        a: (N,) array of vaccination probabilities for each individual
    """
    N = state['N']
    group_id = state['group_id']
    vaccinated = state.get('vaccinated', np.zeros(N))
    supply_today = state.get('supply_today', Ug.sum())

    # Eligible individuals (not yet vaccinated)
    eligible = (vaccinated == 0).astype(float)

    # Get individual-level features
    if isinstance(graph, nx.Graph):
        # NetworkX graph
        degrees = np.array([graph.degree(i) for i in range(N)])
    else:
        # Assume graph has degree_array method
        degrees = graph.degree_array() if hasattr(graph, 'degree_array') else np.ones(N)

    # Individual risk scores (higher for high-risk group)
    if 'risk' in state:
        risk = state['risk']
    else:
        # Default: high-risk group has higher risk
        risk = np.ones(N)
        risk[group_id == 1] = 2.0  # Group 1 is high-risk

    # Compute priority weights based on rule
    if rule == 'degree':
        weights = degrees
    elif rule == 'risk':
        weights = risk
    elif rule == 'degree_risk':
        # Weighted combination of degree and risk
        weights = 0.6 * degrees + 0.4 * risk
    elif rule == 'uniform':
        weights = np.ones(N)
    else:
        raise ValueError(f"Unknown lifting rule: {rule}")

    # Allocate within each group
    a = np.zeros(N)

    for g in range(len(Ug)):
        if Ug[g] <= eps:
            continue

        # Individuals in this group
        idx = np.where(group_id == g)[0]

        # Eligible individuals in this group
        w_g = weights[idx] * eligible[idx]

        # Normalize weights
        w_sum = w_g.sum()
        if w_sum < eps:
            continue

        # Allocate proportionally to weights
        a[idx] = Ug[g] * (w_g / (w_sum + eps))

    # Ensure no one gets more than 1 dose
    a = np.minimum(a, eligible)

    # Scale to respect total supply constraint
    total_allocated = a.sum()
    if total_allocated > supply_today + eps:
        scale = supply_today / (total_allocated + eps)
        a = a * scale

    return a


def compute_macro_quota(
    trajectory: Dict,
    time_step: int
) -> np.ndarray:
    """
    Extract macro-level group allocation from expert trajectory at given time step.

    Args:
        trajectory: Expert trajectory dictionary from simulate_episode_macro
        time_step: Time step index

    Returns:
        Ug: (G,) array of vaccine doses for each group
    """
    U = trajectory['U']  # Shape: (T, G)
    if time_step >= U.shape[0]:
        # Return zeros if beyond trajectory length
        return np.zeros(U.shape[1])

    return U[time_step, :]


def project_feasible(a_t: np.ndarray, state: Dict, eps: float = 1e-8) -> np.ndarray:
    """
    Project individual allocations to feasible region.

    Constraints:
    1. a_t >= 0
    2. Only vaccinate non-vaccinated individuals
    3. Respect group-level quotas (if provided)
    4. Respect total supply constraint
    5. a_t <= 1 (can't give more than one dose)

    Args:
        a_t: (N,) array of vaccination allocations
        state: State dictionary containing constraints
        eps: Small constant for numerical stability

    Returns:
        Feasible allocation a_t
    """
    # 1. Non-negativity
    a_t = np.clip(a_t, 0, None)

    # 2. Only vaccinate eligible individuals
    vaccinated = state.get('vaccinated', np.zeros(len(a_t)))
    a_t = a_t * (vaccinated == 0)

    # 3. Respect group quotas if provided
    if 'macro_quota' in state and 'group_id' in state:
        macro_quota = state['macro_quota']
        group_id = state['group_id']

        for g in range(len(macro_quota)):
            idx = (group_id == g)
            group_allocation = a_t[idx].sum()

            if group_allocation > eps:
                # Scale to match quota
                target = macro_quota[g]
                scale = target / (group_allocation + eps)
                a_t[idx] = a_t[idx] * scale

    # 4. Respect total supply
    if 'supply_today' in state:
        supply = state['supply_today']
        total = a_t.sum()

        if total > supply + eps:
            scale = supply / (total + eps)
            a_t = a_t * scale

    # 5. Maximum 1 dose per person
    a_t = np.minimum(a_t, 1.0)

    return a_t


def create_micro_state_from_macro(
    macro_state: Dict,
    time_step: int,
    graph: nx.Graph,
    group_id: np.ndarray
) -> Dict:
    """
    Create micro-level state dictionary from macro trajectory.

    Args:
        macro_state: Macro trajectory from expert
        time_step: Current time step
        graph: Contact network
        group_id: Group assignment for each individual

    Returns:
        Micro state dictionary compatible with lift_macro_to_micro
    """
    N = len(group_id)
    G = len(np.unique(group_id))

    # Extract group-level states at this time step
    S = macro_state['S'][time_step]  # (G,) susceptible in each group
    V = macro_state['V'][time_step]  # (G,) vaccinated in each group
    U = macro_state['U'][time_step] if time_step < macro_state['U'].shape[0] else np.zeros(G)

    # Approximate individual vaccination status
    vaccinated = np.zeros(N, dtype=int)
    for g in range(G):
        idx = np.where(group_id == g)[0]
        n_group = len(idx)
        n_vax_group = int(V[g])

        if n_vax_group > 0 and n_group > 0:
            # Randomly select vaccinated individuals in proportion
            vax_prob = min(n_vax_group / n_group, 1.0)
            vaccinated[idx] = np.random.binomial(1, vax_prob, size=len(idx))

    state = {
        'N': N,
        'group_id': group_id,
        'vaccinated': vaccinated,
        'supply_today': U.sum(),
        'macro_quota': U,
        'time_step': time_step,
    }

    return state
