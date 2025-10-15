import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from deustorl.common import EpsilonGreedyPolicy, max_policy, evaluate_policy
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.helpers import DiscretizedEnvironmentWrapper
import time
import os

os.system("rm -rf ./activity1/")

env_name = "MountainCarContinuous-v0"
n_steps = 10_000
n_eval_episodes = 1  
tensorboard_log = "./activity1"

env = gym.make(env_name, render_mode="ansi")
env = Monitor(env)

def run_tabular(algo_class, n_bins_obs=10, n_bins_actions=2, lr=0.1):
    """Helper function for tabular algorithms"""
    disc_env = DiscretizedEnvironmentWrapper(gym.make(env_name), n_bins_obs=n_bins_obs, n_bins_actions=n_bins_actions)
    visual_env = DiscretizedEnvironmentWrapper(gym.make(env_name, render_mode="ansi"), n_bins_obs=n_bins_obs, n_bins_actions=n_bins_actions)
    algo = algo_class(disc_env)
    print(f"Testing {algo_class.__name__}")
    start_time = time.time()
    epsilon_policy = EpsilonGreedyPolicy(epsilon=0.1)
    algo.learn(epsilon_policy, n_steps, lr=lr)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))
    
    mean_reward = evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=n_eval_episodes, verbose=False)
    print(f"{algo_class.__name__} Mean reward over {n_eval_episodes} episodes: {mean_reward}")
    return algo

for algorithm in ["SAC", "PPO", "Sarsa", "QLearning", "ExpectedSarsa"]:
    if algorithm == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        model.learn(total_timesteps=n_steps)
        model.save("sac_model")
        mean_reward, std_reward = sb3_evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
        print(f"SAC Mean reward over {n_eval_episodes} episodes: {mean_reward} +/- {std_reward}")
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        model.learn(total_timesteps=n_steps)
        model.save("ppo_model")
        mean_reward, std_reward = sb3_evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
        print(f"PPO Mean reward over {n_eval_episodes} episodes: {mean_reward} +/- {std_reward}")
    elif algorithm == "Sarsa":
        run_tabular(Sarsa)
    elif algorithm == "QLearning":
        run_tabular(QLearning)
    elif algorithm == "ExpectedSarsa":
        run_tabular(ExpectedSarsa)

env.close()
