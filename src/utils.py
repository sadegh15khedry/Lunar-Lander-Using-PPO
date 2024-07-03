import gymnasium as gym
from stable_baselines3 import PPO

def get_env(env_name, render_mode):
   env = gym.make(env_name, render_mode=render_mode)
   return env


def load_model(path):
    model = PPO.load(path)
    return model