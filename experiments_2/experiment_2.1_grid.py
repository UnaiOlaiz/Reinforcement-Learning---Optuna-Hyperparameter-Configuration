import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.helpers import FrozenLakeDenseRewardsWrapper
from deustorl.montecarlo import Montecarlo_FirstVisit
import gymtonic

def test(algo, n_steps= 60000, **kwargs):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)
    print("Avg reward: {:0.4f}, avg steps: {:0.4f}".format(avg_reward,avg_steps))
    print()


env = gym.make("gymtonic/GridTargetSimple-v0", render_mode=None)

seed = 3
random.seed(seed)
env.reset(seed=seed)

os.system("rm -rf ./logs/")

algo = Montecarlo_FirstVisit(env)
print("Testing First Visit Montecarlo")
test(algo, n_steps=20_000)
print(algo.q_table)
input("Press Enter to continue...")

env = gym.make("gymtonic/GridTargetSimple-v0", render_mode='human')

evaluate_policy(env, algo.q_table, max_policy, n_episodes=10, verbose=True)
print(algo.q_table)

env.close()