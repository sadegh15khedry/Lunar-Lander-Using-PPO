# Lunar Lander using Proximal Policy Optimization (PPO)

This repository contains code for training and evaluating a Lunar Lander agent using the Proximal Policy Optimization (PPO) algorithm.

## Prerequisites

- Python 3.x
- TensorFlow
- Stable Baselines3
- Gymnasium

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LunarLander-PPO.git
   cd Lunar-Lander-DeepQ-Learning
   ```

2. Install the required libraries:
   ```bash
   pip install tensorflow stable-baselines3 gymnasium
   ```

## Code Description

### Main Script in lunar_lander_deepQ.py

The main script sets up the environment, trains the model using PPO, evaluates the model, and saves the trained model.

```python
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

# Setting up our environment
env = gym.make("LunarLander-v2", render_mode="human")
env = DummyVecEnv([lambda: env])

# Setting up the model
model = PPO("MlpPolicy", env, verbose = 1)
# Uncomment the following lines to load a pre-trained model
# if LOAD == 1:
#     model = PPO.load("ppo_model.zip")

# Training the model
model.learn(total_timesteps=100000)

# Evaluating our model
evaluate_policy(model, env, n_eval_episodes=10, render=True)

# Saving the model
model.save("ppo_model.zip")

# Closing the environment
env.close()
```

### Random Action Implementation in random_aciton_lunar_lander.py

For comparison, a random action implementation is provided to understand the difference between a trained agent and a random policy.

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np
from collections import deque
import random

episodes = 10
for episode in range(0, episodes):
    state = env.reset()
    done = False
    score = 0
   
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
      
    print(f"Episode: {episode}  Score: {score}")
env.close()
```

## Running the Code

1. To train and evaluate the PPO model, simply run the main script:
   ```bash
   python lunar_lander_deepQ.py
   ```

2. If you want to see the performance of a random action policy:
```bash
   python random_aciton_lunar_lander.py
   ```

## Results

- The script trains a PPO model for the Lunar Lander environment and evaluates its performance.
- The model is saved as `ppo_model.zip` after training.
- The performance is evaluated over 10 episodes, and the average score is printed.

## Further Improvements

- Tune hyperparameters for better performance.
- Experiment with different algorithms available in Stable Baselines3.
- Implement additional evaluation metrics and visualizations.

## Author

- Sadegh Khedery

## Acknowledgment

This project is based on the tutorial by Nicholas Renotte on training a Lunar Lander agent. You can find the tutorial here https://www.youtube.com/watch?v=nRHjymV2PX8&t=551s .
## License

This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details.
