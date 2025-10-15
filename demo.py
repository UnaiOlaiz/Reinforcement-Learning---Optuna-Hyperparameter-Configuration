import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import ale_py
from stable_baselines3 import SAC, PPO # SAC algorithm does not work for this environment

gym.register_envs(ale_py)

env_name = 'ALE/Boxing-v5'
n_steps = 200_000
checkpoint_dir = './checkpoints_boxing'

env = gym.make(env_name, obs_type='grayscale', render_mode='human')

model = PPO('MlpPolicy',
            env=env,
            verbose=1,
            tensorboard_log="./boxing_tensorboard/")

checkpoint_dir = CheckpointCallback(save_freq=20_000, save_path=checkpoint_dir)

model.learn(total_timesteps=n_steps)

model.save("boxing_model_ppo")

env.close()