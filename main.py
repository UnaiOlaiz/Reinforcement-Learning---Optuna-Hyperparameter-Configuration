# Libraries and dependencies we will use for this assignment
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import ale_py
from deustorl.common import evaluate_policy, EpsilonGreedyPolicy, max_policy
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.montecarlo_lrdecay import Montecarlo_FirstVisit_LRDecay
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
from torch.utils.tensorboard import SummaryWriter

# We will use these classes to reduce the complexity of the environment, as it would not be feasible to execute it otherways (too many calculations)
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

# This function will try different hyperparameter configurations
def hyperparameter_analysis(t):
    environment_name = 'ALE/Boxing-v5'
    env = gym.make(
        environment_name,
        obs_type='ram',
        render_mode=None,   # no render para acelerar
        frameskip=4         # frameskip mayor para acelerar
    )
    env = RamFeatureWrapper(env, num_features=8)
    env = DiscretizedObservationWrapper(env, n_bins=4)

    # We first create the tensorboard log directory for this trial
    tensorboard_log_dir = f"./tensorboard_logs/trial_{t.number}/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # --- HP sampling ---
    algorithm_name = t.suggest_categorical(
        "algorithm",
        ["Sarsa", "Qlearning", "ExpectedSarsa", "Montecarlo_FirstVisit_LRDecay"]
    )
    learning_rate   = t.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    discount_factor = t.suggest_float("discount_factor", 0.8, 1.0, step=0.025)
    epsilon         = t.suggest_float("epsilon", 0.0, 0.5, step=0.05)
    lr_decay        = t.suggest_float("lr_decay", 0.85, 1.0, step=0.01)
    lr_episodes_decay = t.suggest_categorical("lr_episodes_decay",[100, 500, 1_000, 5_000, 10_000])

    # Number of steps to train (total en el trial)
    n_steps_total = 100_000

    # Depending on the algorithm, we create the corresponding agent
    if algorithm_name == "Sarsa":
        print("Using SARSA algorithm"); print("Trial number:", t.number)
        algo = Sarsa(env)
    elif algorithm_name == "Qlearning":
        print("Using Q-learning algorithm"); print("Trial number:", t.number)
        algo = QLearning(env)
    elif algorithm_name == "ExpectedSarsa":
        print("Using Expected SARSA algorithm"); print("Trial number:", t.number)
        algo = ExpectedSarsa(env)
    elif algorithm_name == "Montecarlo_FirstVisit_LRDecay":
        print("Using Monte Carlo First Visit LRDecay algorithm"); print("Trial number:", t.number)
        algo = Montecarlo_FirstVisit_LRDecay(env)
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm_name}")

    # Policy
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)

    # Seed por trial (para estabilidad y reproducibilidad en paralelo)
    seed = 88 + t.number
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    # Entrenamiento en chunks + pruning

    steps_done = 0

    try:
        algo.learn(
            epsilon_greedy_policy,
            n_steps=n_steps_total,
            discount_rate=discount_factor,
            lr=learning_rate,
            lrdecay=lr_decay,
            n_episodes_decay=lr_episodes_decay,
            verbose=False
        )

        # evaluación intermedia barata
        #inter_reward, _ = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=5)
        #writer.add_scalar("eval/intermediate_reward", inter_reward, steps_done)

        # report a Optuna y permite podar
        # t.report(inter_reward, step=steps_done)
        # if t.should_prune():
            # raise optuna.TrialPruned()

        # evaluación final del trial (más estable que 5, menos caro que 100)
        avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=20)
        writer.add_scalar("eval/avg_reward", avg_reward, t.number)
        writer.add_scalar("eval/avg_steps",  avg_steps,  t.number)

        # hparams resumidos
        writer.add_hparams({
            "algorithm": algorithm_name,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "lr_decay": lr_decay,
            "lr_episodes_decay": lr_episodes_decay
        }, {"avg_reward": avg_reward})

        return avg_reward
    finally:
        writer.close()
        env.close()

if __name__ == "__main__": 
    # Removal of old logs and directories
    os.system("rm -rf ./tensorboard_logs/")
    os.system("rm -rf ./boxing_db/")
    os.system("rm -rf ./boxing_optuna.db")
    os.system("rm -rf ./Reinforcement-Learning/tensorboard_logs/")
    os.system("rm -rf ./boxing_optuna/")
    os.system("rm -rf ./boxing_logs/")
    os.system("mkdir -p ./boxing_optuna/")

    # Random seed
    seed = 88
    random.seed(seed)

    # Optuna study setup
    boxing_db = f"sqlite:///boxing_optuna/boxing_optuna.db"
    environment_study_name = "boxing"
    full_study_dir_path = f"boxing_optuna/{environment_study_name}"
    sampler = TPESampler(seed=seed)
    #pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(
        sampler=sampler,
        #pruner=pruner,
        direction='maximize',
        study_name=environment_study_name,
        storage=boxing_db,
        load_if_exists=True
    )
    
    # Number of trials for the hyperparameter search
    trials_per_alg = 50
    total_trials = 4 * trials_per_alg  # 200 trials en total

    # Optuna hyperparameter search time
    print("Starting the optuna hyperparameter search for environment: ALE/Boxing-v5")

    # Paraleliza sin saturar (ajusta si notas CPU al 100%)
    n_workers = min(max(1, os.cpu_count() - 1), 8)
    study.optimize(hyperparameter_analysis, n_trials=total_trials, n_jobs=n_workers)

    # Save the best trial hyperparameters
    best_search = study.best_trial
    best_search_params = json.dumps(best_search.params, sort_keys=True, indent=4)

    # We create the study directory
    os.system(f"mkdir -p {full_study_dir_path}")

    # We keep the best trial parameters in a json file
    best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
    best_trial_file.write(best_search_params)
    best_trial_file.close()

    # Generate the important figures of the results
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{full_study_dir_path}/optimization_history.html")
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(f"{full_study_dir_path}/contour.html")
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{full_study_dir_path}/slice.html")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{full_study_dir_path}/param_importances.html")