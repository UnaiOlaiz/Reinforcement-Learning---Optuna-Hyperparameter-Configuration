import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import ale_py
from deustorl.common import evaluate_policy, EpsilonGreedyPolicy, max_policy
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
import time
import os
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import cv2
import numpy as np
from deustorl.helpers import DiscretizedObservationWrapper
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import random
import json


class DownsampleObservationWrapper(ObservationWrapper):
    def __init__(self, env, new_shape=(8, 8)):
        super().__init__(env)
        self.new_shape = new_shape
        self.observation_space = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        return cv2.resize(obs, self.new_shape, interpolation=cv2.INTER_AREA)

class RamFeatureWrapper(ObservationWrapper):
    def __init__(self, env, num_features=8):
        super().__init__(env)
        self.num_features = num_features
        self.observation_space = Box(low=0, high=255, shape=(num_features,), dtype=np.uint8)

    def observation(self, obs):
        return obs[:self.num_features]

def hyperparameter_analysis(t):
    algorithm_name = t.suggest_categorical("algorithm_name", ["Sarsa", "Qlearning", "Expectedsarsa"])
    learning_rate = t.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    discount_factor = t.suggest_float("discount_factor", 0.8, 1.0, step=0.05)
    epsilon = t.suggest_float("epsilon", 0.0, 0.4, step=0.05)
    lr_decay = t.suggest_float("lr_decay", 0.9 , 1.0,  step=0.01)
    lr_episodes_decay = t.suggest_categorical("lr_episodes_decay",[100, 1_000, 10_000]) 

    n_steps = 100_000

    if algorithm_name == "Sarsa":
        algo = Sarsa(env)
    elif algorithm_name == "Qlearning":
        algo = QLearning(env)
    elif algorithm_name == "Expectedsarsa":
        algo = ExpectedSarsa(env)

    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    algo.learn(epsilon_greedy_policy, n_steps=n_steps, discount_rate=discount_factor, lr=learning_rate, lrdecay=lr_decay, n_episodes_decay=lr_episodes_decay)

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100)

    return avg_reward

if __name__ == "__main__": 


    environment_name = 'ALE/Boxing-v5'
    env = gym.make(environment_name, obs_type='ram', render_mode='rgb_array', frameskip=1)
    env = RamFeatureWrapper(env, num_features=8)
    env = DiscretizedObservationWrapper(env, n_bins=4)


    seed = 88
    random.seed(seed)

    n_steps = 100_000

    os.system("rm -rf ./boxing_logs/")
    os.system("mkdir -p ./boxing_optuna/")
    boxing_db = f"sqlite:///boxing_optuna/boxing_optuna.db"
    environment_study_name = "boxing"
    full_study_dir_path = f"boxing_optuna/{environment_study_name}"
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction='maximize', study_name=environment_study_name, storage=boxing_db, load_if_exists=True)
    trials = 10

    # Optuna hyperparameter search time
    print(f"Starting the optuna hyperparameter search for environment: {environment_name}")
    study.optimize(hyperparameter_analysis, n_trials=trials)

    env.close()

    best_search = study.best_trial
    best_search_params = json.dumps(best_search.params, sort_keys=True, indent=4)

    os.system(f"mkdir -p {full_study_dir_path}")
    best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
    best_trial_file.write(best_search_params)
    best_trial_file.close()

    # Generate the improtant figures of the results
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{full_study_dir_path}/optimization_history.html")
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(f"{full_study_dir_path}/contour.html")
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{full_study_dir_path}/slice.html")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{full_study_dir_path}/param_importances.html")