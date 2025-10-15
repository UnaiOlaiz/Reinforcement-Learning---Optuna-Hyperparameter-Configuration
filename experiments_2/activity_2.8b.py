import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa

def test(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)


os.system("rm -rf ./logs/")

env_name = "Taxi-v3" 
env = gym.make(env_name)
visual_env = gym.make(env_name, render_mode='human')


seed = 3
random.seed(seed)
env.reset(seed=seed)

n_steps = 200_000

algo = Sarsa(env)
print("Testing SARSA")
test(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = QLearning(env)
print("Testing Q-Learning")
test(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)

algo = ExpectedSarsa(env)
print("Testing Expected SARSA")
test(algo, n_steps=n_steps, lr=0.1)
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=10, verbose=False)


