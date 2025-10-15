import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration parameters
env_name = 'MountainCarContinuous-v0' # environment for SAC
n_steps = 200_000 # SAC usually requires more training steps
checkpoint_dir = './checkpoints_sac/' # directory to save checkpoints

# Creation of the environment
env = gym.make(env_name, render_mode="ansi")

# Creation of the SAC (SB3 uses terminated/truncated)
model = SAC('MlpPolicy',  # if the environment uses numbers -> 'MlpPolicy', 'CnnPolicy' for images
            env, 
            verbose=1, # verbose=1 for info messages (0 no output)
            tensorboard_log="./sac_mountaincar_tensorboard/")

# Each 20_000 steps of the training we save a checkpoint, just in case of battery run out, connection loss, ...
checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path=checkpoint_dir, name_prefix="sac_mountaincar")

# Train the model 
model.learn(total_timesteps=n_steps)

# Save of the final model
model.save("sac_mountaincar_model")

# Closure of the environment
env.close()



