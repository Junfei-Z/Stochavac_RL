# main.py

from graph_model import create_contact_graph
from epidemic_env import EpidemicEnv
from policy_baseline import baseline_policy
from config import EPISODE_LENGTH

def run_baseline():
    G = create_contact_graph()
    env = EpidemicEnv(G)
    state = env.reset()
    done = False
    print(f"Initial State: {state}")
    while not done:
        action = baseline_policy(state, G)
        state, reward, done, _ = env.step(action)
        print(f"Time {env.time} | Reward: {reward} | State: {state}")

if __name__ == "__main__":
    run_baseline()
