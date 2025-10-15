# Libraries we will use
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import ale_py
from stable_baselines3 import SAC, PPO # SAC algorithm does not work for this environment
from deustorl.common import evaluate_policy, EpsilonGreedyPolicy, max_policy
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
import time
import os

os.system("rm -rf ./tensorboard_logs/")
os.system("rm -rf ./checkpoints/")
os.system("rm -rf ./models/")
os.makedirs("./tensorboard_logs/", exist_ok=True)
os.makedirs("./checkpoints/", exist_ok=True)
os.makedirs("./models/", exist_ok=True)

gym.register_envs(ale_py)
env_name = 'ALE/Boxing-v5'
n_steps = 200_000

env = gym.make(env_name, obs_type='grayscale', render_mode='rgb_array')

# List of algorithms we will use
algoritms = ['PPO', 'SAC', 'SARSA', 'Qlearning', 'Expectedsarsa']

# Function we will use to solve the last 3 algorithms
def test(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)

for algo in algoritms:
    if algo == "PPO":
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"./tensorboard_logs/tensorboard_{algo.lower()}/")
        model.learn(n_steps)
        checkpoint_dir = CheckpointCallback(save_freq=20_000, save_path=f"./checkpoints/checkpoint_{algo.lower()}/",)
        model.save(f"./models/boxing_model_{algo.lower()}")
    elif algo == "SARSA":
        algo = Sarsa(env)
        test(algo, n_steps=n_steps, lr=0.01)
        checkpoint_dir = CheckpointCallback(save_freq=20_000, save_path=f"./checkpoints/checkpoint_{algo.lower()}/")
        evaluate_policy(env, algo.q_table, max_policy, n_episodes=10, verbose=False)
    elif algo == "Qlearning":   
        algo = QLearning(env)
        test(algo, n_steps=n_steps, lr=0.01)
        checkpoint_dir = CheckpointCallback(save_freq=20_000, save_path=f"./checkpoints/checkpoint_{algo.lower()}/")
        evaluate_policy(env, algo.q_table, max_policy, n_episodes=10, verbose=False)
    elif algo == "Expectedsarsa":
        algo = ExpectedSarsa(env)
        test(algo, n_steps=n_steps, lr=0.01)
        checkpoint_dir = CheckpointCallback(save_freq=20_000, save_path=f"./checkpoints/checkpoint_{algo.lower()}/")
        evaluate_policy(env, algo.q_table, max_policy, n_episodes=10, verbose=False)

env.close()

# Creo que no funciona el SAC para este environment
'''
    elif algo == "SAC":
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"./tensorboard_logs/tensorboard_{algo.lower()}/")
        model.learn(n_steps)
        checkpoint_dir = CheckpointCallback(save_freq=20_000, save_path=f"./checkpoints/checkpoint_{algo.lower()}/")
        model.save(f"./models/boxing_model_{algo.lower()}")
    '''
