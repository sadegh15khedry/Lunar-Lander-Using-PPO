# Lunar Lander Using Proximal Policy Optimization (PPO)

This repository contains code for training and evaluating a Lunar Lander agent using the Proximal Policy Optimization (PPO) algorithm.


## Table of Contents


- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
- [Model Performance](#model-performance)
- [Installation Guide](#installation-guide)
- [Further Improvements](#further-improvements)
- [Acknowledgments](#acknowledgments)
- [License](#license)


## Directory Structure
```
├── src
│ ├── utils.py
│ ├── lunar_lander_training.py
│ └── lunar_lander_evaluation.py
├── notebooks
│ ├── lunar_lander_training.ipynb
│ └── lunar_lander_evaluation.ipynb
├── environment.yml
└── README.md
```


## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `lunar_lander_training.py` : Functions for training the lunar lander.
- `lunar_lander_evaluation.py` : Functions for evaluating the lunar lander.
- `lunar_lander_training.ipynb`: Notebook for lunar lander training.
- `lunar_lander_evaluation.ipynb`: Notebook for lunar lander evaluation.


## Model Performance

This section will be added
  
## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/Lunar-Lander-Using-PPO.git
    cd Lunar-Lander-Using-PPO
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate lunar-lander
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```

## Further Improvements

- Tune hyperparameters for better performance.
- Experiment with different algorithms available in Stable Baselines3.
- Implement additional evaluation metrics and visualizations.


## Acknowledgment

This project is based on the tutorial by Nicholas Renotte on training a Lunar Lander agent. You can find the tutorial here https://www.youtube.com/watch?v=nRHjymV2PX8&t=551s .

## License

This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details.












