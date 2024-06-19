import gym
from gym.envs import box2d
import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym

#setting up gymnasium
env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()
episodes = 10
for episode in range(0, episodes):
   state = env.reset()
   done = False
   score = 0
   
   while not done:
      env.render()
      action = env.action_space.sample()
      # n_state, reward, done, info = env.step(action)
      obs, reward, done, truncated, info = env.step(action)
      score += reward
      
   print(f"Episode: {episode}  Score: {score}")
env.close()





# # Set up the environment
# env = gym.make("LunarLander-v2", render_mode= "human")

# episodes = 10
# for episode in range(1, episodes+1):
#   state = env.reset()
#   done = False 
#   score = 0

#   while not done:
#     action = random.choice([0, 1, 2])
#     _, reward, done, _ = env.step(action)
#     score += reward

#   print(f"episode: {episode} , score: {score}")

# env.close()

