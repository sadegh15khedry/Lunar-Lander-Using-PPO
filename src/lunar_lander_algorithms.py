import stable_baselines3 as sb3
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def run_random_algorithm(env, episodes):
    for episode in range(0, episodes):
      info = env.reset()
      done = False
      score = 0
   
      while not done:
         env.render()
         action = env.action_space.sample()
         try:
            obs, reward, done, info = env.step(action)
         except ValueError:
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
         score += reward
      
      print(f"Episode: {episode}  Score: {score}")
    env.close()
    
    
    
    
def run_PPO_algorithm(env, total_timesteps, load=False):
    #setting up the model
   model = PPO("MlpPolicy", env, verbose = 1)
   
   if load == True:
      model = sb3.PPO.load("ppo_model.zip")

   #training the model
   model.learn(total_timesteps=total_timesteps)

   #evalutating our model
   evaluate_policy(model, env,  n_eval_episodes=10, render=True)

   #Saving the model
   model.save("ppo_model.zip")

   #closing the environment
   env.close()
   
   
def get_env(env_name, render_mode):
   env = gym.make(env_name, render_mode=render_mode)
   # env = DummyVecEnv([lambda: env])
   return env