# rl_agent.py

from epidemic_env_gym import EpidemicGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def train_agent(total_timesteps=100_000):
    env = make_vec_env(EpidemicGymWrapper, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model
