"""
Expert Trajectory Generation from ODE-based ProtectorPrevent Model

This script wraps the ProtectorPrevent ODE model to generate macro-level
expert trajectories for training the diffusion model.
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Add third_party to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/ProtectorPrevent/Simulation'))

# Use fixed version of runSim to handle 3-group constraint
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from expert.run_sim_wrapper import runSim_fixed as runSim
from helper_functions import scale_contacts


def simulate_episode_macro(seed: int, scenario: dict) -> dict:
    """
    Simulate a single episode using the ODE expert policy.

    Args:
        seed: Random seed for reproducibility
        scenario: Dictionary containing:
            - r_0: Basic reproduction number
            - vax_rate: Vaccination rate (fraction of population per day)
            - init_infections: Initial number of infections
            - strategy: One of ['uniform', 'high risk', 'high contact']
            - f_high_risk: Fraction of high-risk population
            - f_high_contact: Fraction of high-contact population
            - vax_efficacy: Vaccine efficacy
            - npi_efficacy: Non-pharmaceutical intervention efficacy
            - variant_param: Variant transmission multiplier

    Returns:
        Dictionary containing:
            - S, E, P, A, I, L, H, V, R, D: Arrays of shape (T+1, G) for each state
            - U: Array of shape (T, G) for vaccine allocation
            - vax_rates: Array of shape (T, G) for vaccination rates
            - meta: Metadata about the simulation
    """
    np.random.seed(seed)

    # Extract parameters
    r_0 = scenario.get('r_0', 2.0)
    vax_rate = scenario.get('vax_rate', 0.005)
    init_infections = scenario.get('init_infections', 15)
    strategy = scenario.get('strategy', 'high risk')
    f_high_risk = scenario.get('f_high_risk', 0.168)
    f_high_contact = scenario.get('f_high_contact', 0.15)
    vax_effi = scenario.get('vax_efficacy', 0.8)
    npi_effi = scenario.get('npi_efficacy', 0.3)
    var_param = scenario.get('variant_param', 1.0)

    # Population setup
    f_base = 1 - f_high_risk - f_high_contact
    N = 2000

    # Initial state: [S, E, P, A, I, L, H, V, R, D] for each group
    init_state = np.array([
        [f_base*(N-init_infections), f_base*init_infections, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [f_high_risk*(N-init_infections), f_high_risk*init_infections, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [f_high_contact*(N-init_infections), f_high_contact*init_infections, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    pop_sizes = np.sum(init_state, axis=1)

    # Contact matrix (USA default)
    contact_matrix = np.array([
        [0.165, 0.1, 0.175],
        [0.1, 0, 0.002],
        [0.175, 0.002, 0.132]
    ])

    # State transition durations
    state_lengths = {
        "E": [4.0, 4.0, 4.0],
        "P": [2.0, 2.0, 2.0],
        "A": [10.0, 10.0, 10.0],
        "I": [3.0, 3.0, 3.0],
        "L": [3.0, 3.0, 3.0],
        "H": [11.0, 11.0, 11.0],
    }

    # Transition probabilities
    trans_probabilities = {
        "symp": [0.4, 0.8, 0.4],
        "hosp": [0.1, 0.3, 0.1],
        "dec": [0.01, 0.1, 0.01],
    }

    transmission_rates = np.array([0.4, 0.8, 0.4])

    vax_efficacy = np.array([vax_effi, vax_effi, vax_effi])
    npi_efficacy = np.array([npi_effi, npi_effi, npi_effi])

    variant = {
        "exp": [var_param, var_param, var_param],
        "symp": [var_param, var_param, var_param],
        "hosp": [var_param, var_param, var_param],
        "dec": [var_param, var_param, var_param]
    }

    # Run simulation
    sim_results, vax_rates, vax_nums, total_hosps, total_infs, cumul_infs, cumul_hosps = runSim(
        init_state, contact_matrix, state_lengths, trans_probabilities,
        transmission_rates, vax_rate, vax_efficacy, npi_efficacy, variant, strategy, r_0
    )

    # Extract trajectories (shape: [T+1, G, 10])
    T = sim_results.shape[0] - 1  # Number of time steps
    G = 3  # Number of groups

    # Note: vax_rates and vax_nums have shape (T, 4) but we only use first 3 groups
    vax_rates = vax_rates[:, :3]
    vax_nums = vax_nums[:, :3]

    # Return structured trajectory
    trajectory = {
        'S': sim_results[:, :, 0],  # Susceptible
        'E': sim_results[:, :, 1],  # Exposed
        'P': sim_results[:, :, 2],  # Pre-symptomatic
        'A': sim_results[:, :, 3],  # Asymptomatic
        'I': sim_results[:, :, 4],  # Infected (symptomatic)
        'L': sim_results[:, :, 5],  # Light symptoms
        'H': sim_results[:, :, 6],  # Hospitalized
        'V': sim_results[:, :, 7],  # Vaccinated
        'R': sim_results[:, :, 8],  # Recovered
        'D': sim_results[:, :, 9],  # Dead
        'U': vax_nums,  # Vaccine allocation [T, G]
        'vax_rates': vax_rates,  # Vaccination rates [T, G]
        'meta': {
            'r_0': r_0,
            'vax_rate': vax_rate,
            'init_infections': init_infections,
            'strategy': strategy,
            'f_high_risk': f_high_risk,
            'f_high_contact': f_high_contact,
            'vax_efficacy': vax_effi,
            'npi_efficacy': npi_effi,
            'variant_param': var_param,
            'N': N,
            'T': T,
            'G': G,
            'total_hosps': total_hosps,
            'total_infs': total_infs,
            'cumul_infs': cumul_infs,
            'cumul_hosps': cumul_hosps,
            'seed': seed
        }
    }

    return trajectory


def generate_scenarios(n_scenarios=1000):
    """
    Generate diverse scenarios by varying key parameters.
    """
    scenarios = []

    # Parameter ranges
    r_0_vals = [1.5, 2.0, 2.5, 3.0]
    vax_rates = [0.002, 0.005, 0.008]
    init_infections_vals = [5, 10, 15, 20]
    strategies = ['uniform', 'high risk', 'high contact']
    f_high_contact_vals = [0.10, 0.15, 0.20]
    vax_efficacy_vals = [0.7, 0.8, 0.9]
    npi_efficacy_vals = [0.0, 0.3, 0.6]
    variant_params = [1.0, 1.2, 1.4]

    for i in range(n_scenarios):
        scenario = {
            'r_0': np.random.choice(r_0_vals),
            'vax_rate': np.random.choice(vax_rates),
            'init_infections': np.random.choice(init_infections_vals),
            'strategy': np.random.choice(strategies),
            'f_high_risk': 0.168,  # Fixed for USA
            'f_high_contact': np.random.choice(f_high_contact_vals),
            'vax_efficacy': np.random.choice(vax_efficacy_vals),
            'npi_efficacy': np.random.choice(npi_efficacy_vals),
            'variant_param': np.random.choice(variant_params),
        }
        scenarios.append(scenario)

    return scenarios


def main():
    """Generate expert trajectories and save to disk."""

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'macro_expert'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate scenarios
    print("Generating scenarios...")
    n_trajectories = 1000
    scenarios = generate_scenarios(n_trajectories)

    # Generate trajectories
    print(f"Generating {n_trajectories} expert trajectories...")
    trajectories = []

    for i, scenario in enumerate(tqdm(scenarios)):
        try:
            traj = simulate_episode_macro(seed=i, scenario=scenario)
            trajectories.append(traj)

            # Save intermediate results every 100 trajectories
            if (i + 1) % 100 == 0:
                intermediate_file = output_dir / f'trajectories_batch_{i//100}.pkl'
                with open(intermediate_file, 'wb') as f:
                    batch_start = (i // 100) * 100
                    pickle.dump(trajectories[batch_start:i+1], f)
                print(f"Saved batch {i//100 + 1} ({i+1} trajectories)")

        except Exception as e:
            print(f"Error in trajectory {i}: {e}")
            continue

    # Save all trajectories
    output_file = output_dir / 'expert_trajectories.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"✓ Saved {len(trajectories)} trajectories to {output_file}")

    # Save summary statistics
    summary = {
        'n_trajectories': len(trajectories),
        'scenarios': scenarios[:10],  # Save first 10 as examples
        'avg_deaths': np.mean([t['D'][-1].sum() for t in trajectories]),
        'avg_infections': np.mean([t['meta']['cumul_infs'] for t in trajectories]),
        'strategies': {s: sum(1 for t in trajectories if t['meta']['strategy'] == s)
                      for s in ['uniform', 'high risk', 'high contact']}
    }

    summary_file = output_dir / 'summary.pkl'
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)

    print("\n=== Summary Statistics ===")
    print(f"Total trajectories: {summary['n_trajectories']}")
    print(f"Average deaths: {summary['avg_deaths']:.2f}")
    print(f"Average cumulative infections: {summary['avg_infections']:.2f}")
    print(f"Strategy distribution: {summary['strategies']}")
    print(f"\n✓ M1 Complete: Expert trajectories saved to {output_dir}")


if __name__ == '__main__':
    main()
