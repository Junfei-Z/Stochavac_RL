"""
Wrapper for ProtectorPrevent's runSim to fix the 4-group issue.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'third_party' / 'ProtectorPrevent' / 'Simulation'))

from helper_functions import scale_contacts, epi_step


def vax_step_fixed(N, current_state, vax_rate, vax_efficacy, strategy, timestep):
    """
    Fixed version of vax_step that works with 3 groups.
    """
    next_state = current_state
    vax_remain = vax_rate * N * timestep

    num_clusters = current_state.shape[0]  # Should be 3

    switcher = {
        "high risk": [1, 2, 0],
        "high contact": [2, 1, 0],
        "uniform": [0, 1, 2],
        "none": [0, 1, 2]
    }

    order = switcher[strategy]
    rates = np.zeros((1, num_clusters))
    nums = np.zeros((1, num_clusters))

    if strategy != "uniform" and strategy != "none":
        for cluster in order:
            if cluster >= num_clusters:
                continue

            if current_state[cluster, 0] >= vax_remain:
                next_state[cluster, 0] -= vax_remain
                next_state[cluster, 7] += vax_remain
                rates[0, cluster] = vax_remain / (current_state[cluster, 0] + 1e-8)
                nums[0, cluster] = vax_remain
                vax_remain = 0
                break
            else:
                next_state[cluster, 0] = 0
                next_state[cluster, 7] += current_state[cluster, 0]
                rates[0, cluster] = 1.0
                nums[0, cluster] = current_state[cluster, 0]
                vax_remain -= current_state[cluster, 0]

    elif strategy == "none":
        pass  # rates and nums already zeros

    else:  # uniform
        total_susceptible = sum(current_state[i, 0] for i in range(num_clusters))
        if total_susceptible > 0:
            rate = vax_remain / total_susceptible
            for i in range(num_clusters):
                rates[0, i] = rate
                nums[0, i] = rate * current_state[i, 0]
                next_state[i, 0] -= nums[0, i]
                next_state[i, 7] += nums[0, i]

    return next_state, rates, nums


def runSim_fixed(init_state, contact_matrix, state_lengths, trans_probabilities,
                 transmission_rates, vax_rate, vax_efficacy, npi_efficacy, variant,
                 strategy, r_0=None):
    """
    Fixed version of runSim using the corrected vax_step.
    """
    N = sum(sum(init_state))
    pop_sizes = np.sum(init_state, axis=1)

    C = scale_contacts(pop_sizes, N, contact_matrix, state_lengths, transmission_rates,
                      trans_probabilities, npi_efficacy, variant, r_0)

    T = 600.0
    timestep = 0.01

    num_clusters = init_state.shape[0]
    num_states = init_state.shape[1]

    sim_results = np.tile(0.0, (round(T/timestep)+1, num_clusters, num_states))
    sim_results[0, :, :] = init_state
    vax_rates = np.tile(0.0, (round(T/timestep)+1, num_clusters))
    vax_nums = np.tile(0.0, (round(T/timestep)+1, num_clusters))
    total_infs = [0] * num_clusters
    total_hosps = [0] * num_clusters
    cumul_infs = 0
    cumul_hosps = 0

    for i in range(1, round(T/timestep)+1, 1):
        sim_results[i, :, :], new_hosps, new_infs = epi_step(
            pop_sizes, sim_results[i-1], C, state_lengths, trans_probabilities,
            transmission_rates, variant, vax_efficacy, timestep, r_0
        )

        total_infs = [sum(x) for x in zip(total_infs, new_infs)]
        total_hosps = [sum(x) for x in zip(total_hosps, new_hosps)]
        cumul_infs += sum(sim_results[i, :, 4]) * timestep
        cumul_hosps += sum(sim_results[i, :, 6]) * timestep

        sim_results[i, :, :], vax_rates[i, :], vax_nums[i, :] = vax_step_fixed(
            N, sim_results[i], vax_rate, vax_efficacy, strategy, timestep
        )

    return sim_results, vax_rates, vax_nums, total_hosps, total_infs, cumul_infs, cumul_hosps
