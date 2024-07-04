import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


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
    

    
    
def run_PPO_algorithm(env, total_timesteps, load=False, log_interval=10, tensorboard_log="./ppo_lunarlander_tensorboard/"):
    #setting up the model
   model = PPO("MlpPolicy", env,  verbose=1)
   # model.set_parameters({
   #  'learning_rate': 1e-4,
   #  'n_steps': 4096,
   #  'batch_size': 64,
   #  'n_epochs': 10,
   #  'gamma': 0.99,
   #  'gae_lambda': 0.95,
   #  'clip_range': 0.2,
   #  'ent_coef': 0.01
   # })
   
   # Loading the model if necessary
   if load == True:
      model = sb3.PPO.load("ppo_model.zip")
      
   # Optionally, update hyperparameters if needed
   

   #training the model
   model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

   #evalutating our model
   evaluate_policy(model, env,  n_eval_episodes=10, render=True)

   #Saving the model
   model.save("../models/ppo_model.zip")

   #closing the environment
   env.close()
   
   
