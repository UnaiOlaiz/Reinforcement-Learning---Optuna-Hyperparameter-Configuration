import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os

os.system("rm -rf ./logs/")

env_name = 'LunarLander-v3' 
n_steps = 3_000
checkpoint_dir = './checkpoints/'
n_eval_episodes = 10 # 10 different trials to evaluate the agent

env = gym.make(env_name, render_mode="ansi")
env = Monitor(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./lunar_lander_ppo/")

checkpoint_callback = CheckpointCallback(save_freq=1_000, save_path=checkpoint_dir, name_prefix="ppo_lunar")

model.learn(total_timesteps=n_steps, callback=checkpoint_callback)

model.save("ppo_lunar_model")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes, render=False)

print(f"Mean reward over {n_eval_episodes} episodes: {mean_reward} +/- {std_reward}")

env.close()