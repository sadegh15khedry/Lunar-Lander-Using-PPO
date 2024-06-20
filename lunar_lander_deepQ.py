import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

LOAD = 0


#setting up our environment
env = gym.make("LunarLander-v2", render_mode="human")
env = DummyVecEnv([lambda: env])

#setting up the model
model = PPO("MlpPolicy", env, verbose = 1)
# if (LOAD == 1)
#     model = sb3.PPO.load("ppo_model.zip")

#training the model
model.learn(total_timesteps=100000)

#evalutating our model
evaluate_policy(model, env,  n_eval_episodes=10, render=True)

#Saving the model
model.save("ppo_model.zip")

#closing the environment
env.close()



##Random aciton implementation
# import numpy as np
# from collections import deque
# import random

# episodes = 10
# for episode in range(0, episodes):
#    state = env.reset()
#    done = False
#    score = 0
   
#    while not done:
#       env.render()
#       action = env.action_space.sample()
#       obs, reward, done, truncated, info = env.step(action)
#       score += reward
      
#    print(f"Episode: {episode}  Score: {score}")
# env.close()






