import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
def get_env(env_name, render_mode):
   env = gym.make(env_name, render_mode=render_mode)
   return env


def load_model(path):
    model = PPO.load(path)
    return model
 
def calculate_metrics(episode_rewards,successes, episodes, episode_lengths ):
   average_return = np.mean(episode_rewards)
   std_return = np.std(episode_rewards)
   success_rate = successes / episodes
   average_length = np.mean(episode_lengths)

   print("\nEvaluation Metrics:")
   print(f"Average Return: {average_return}")
   print(f"Standard Deviation of Return: {std_return}")
   print(f"Success Rate: {success_rate * 100}%")
   print(f"Average Episode Length: {average_length}")