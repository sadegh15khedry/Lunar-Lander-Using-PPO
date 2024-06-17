import gym
import numpy as np
from collections import deque
import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam



# Set up the environment
env = gym.make("LunarLander-v2", render_mode= "human")

episodes = 10
for episode in range(1, episodes+1):
  state = env.reset()
  done = False 
  score = 0

  while not done:
    action = random.choice([0, 1, 2])
    _, reward, done, _ = env.step(action)
    score += reward

  print(f"episode: {episode} , score: {score}")

env.close()






