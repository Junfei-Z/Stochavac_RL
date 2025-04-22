# main.py

import argparse
from policy_baseline import baseline_policy
from graph_model import create_contact_graph
from epidemic_env import EpidemicEnv
from rl_agent import train_agent

def run_baseline():
    G = create_contact_graph()
    env = EpidemicEnv(G)
    state = env.reset()
    done = False
    while not done:
        action = baseline_policy(state, G)
        state, reward, done, _ = env.step(action)
        print(f"Time {env.time} | Reward: {reward} | State: {state}")

def run_rl():
    train_agent()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="baseline", choices=["baseline", "rl"])
    args = parser.parse_args()

    if args.policy == "baseline":
        run_baseline()
    elif args.policy == "rl":
        run_rl()

