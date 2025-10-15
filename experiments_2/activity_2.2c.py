import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.montecarlo import Montecarlo_FirstVisit

def test(algo, n_steps=60000, **kwargs):
    start_time = time.time()
    epsilon = 0.4
    n_phases = 4
    for n in range(n_phases):
        epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
        algo.learn(epsilon_greedy_policy, n_steps//n_phases, **kwargs)
        epsilon /= 2
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))
    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)
    

os.system("rm -rf ./logs/")

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="ansi")
visual_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")

seed = 3
random.seed(seed)
env.reset(seed=seed)

n_rounds = 1
n_steps_per_round = 200_000

print("Testing First Visit Montecarlo")
total_reward = 0
for _ in range(n_rounds):
    algo = Montecarlo_FirstVisit(env)
    avg_reward, _ = test(algo, n_steps=n_steps_per_round)
    total_reward += avg_reward

print("Average reward over {} rounds: {:.4f}".format(n_rounds, total_reward/n_rounds))
input("Press Enter to continue...")
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=3, verbose=False)

print("Testing First Visit Montecarlo with discount_rate=0.95")
total_reward = 0
for _ in range(n_rounds):
    algo = Montecarlo_FirstVisit(env)
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, discount_rate=0.95)
    total_reward += avg_reward

print("Average reward over {} rounds: {:.4f}".format(n_rounds, total_reward/n_rounds))
input("Press Enter to continue...")
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=3, verbose=False)

print("Testing First Visit Montecarlo with discount_rate=0.90")
total_reward = 0
for _ in range(n_rounds):
    algo = Montecarlo_FirstVisit(env)
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, discount_rate=0.90)
    total_reward += avg_reward

print("Average reward over {} rounds: {:.4f}".format(n_rounds, total_reward/n_rounds))
input("Press Enter to continue...")
evaluate_policy(visual_env, algo.q_table, max_policy, n_episodes=3, verbose=False)
