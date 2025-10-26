# Reinforcement Learning – Optuna Hyperparameter Configuration

A **reinforcement learning** project focused on **hyperparameter optimization** using **Optuna**.  
The goal is to train intelligent agents in the *Atari Boxing* environment and analyze how different hyperparameter configurations affect performance using different RL algorithms.

---

## Table of Contents

- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [License](#license)

---

## Description

This repository is part of the **Reinforcement Learning** course at the **University of Deusto**.  
It explores the use of **Optuna**, an automatic hyperparameter optimization framework, applied to various RL algorithms such as **SARSA**, **Q-Learning**, and **Expected SARSA**.

The project uses the **Atari Boxing** environment to evaluate how algorithm performance evolves under different learning conditions.  
By combining Optuna’s *TPE Sampler* with multi-core parallelization, the search space is explored efficiently to identify the best configurations.

## Prerequisites

Before running the project, make sure you have installed:

- **Python 3.8+**
- **Git**
- (Optional) **Conda** or **virtualenv** for managing environments
- (Optional) **CUDA-enabled GPU** for faster training (if using deep RL extensions)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/UnaiOlaiz/Reinforcement-Learning---Optuna-Hyperparameter-Configuration.git
cd Reinforcement-Learning---Optuna-Hyperparameter-Configuration
```
### 2. Creation of the Conda environment
```bash
conda create -n rl python=3.10
conda activate rl
pip install -r requirements.txt
``` 

### 3. Project Structure
``` 
Reinforcement-Learning---Optuna-Hyperparameter-Configuration/
├── boxing_optuna/                 # Core RL training scripts or algorithm implementations
├── boxing_top_3_trials.txt        # Top 3 hyperparameter combinations results
├── deustorl/                      # University-specific scripts, helpers, or modules
├── logs/                          # Log files generated during experiments
├── requirements.txt               # Python dependencies for the project
├── src/                           # Source code and experiment scripts
│   ├── get_best_trials.py
│   ├── main.py
│   ├── top_3.py
│   └── (other source files, wrappers, etc.)
├── tensorboard_logs/              # TensorBoard logs for full experiment runs
└── tensorboard_logs_top_3/        # TensorBoard logs for the top 3 hyperparameter runs
```

### 4. Licenses
This project is open source and available for educational and investigation purposes.

