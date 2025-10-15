import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Resource Collector called custom environment
# The agent collects resources that appear randomly
# 2 actions: L or R
# Collecting a resource = + 1 reward
# Running into an empty grid = - 0.1 reward
# 1 episode = 20 steps

class ResourceCollectorEnv(gym.Env):
    def __init__(self):
        super(ResourceCollectorEnv, self).__init__()
        self.grid_size = 5
        self.max_steps = 20
        self.current_step = 0

        self.action_space = spaces.Discrete(2) # 2 discrete actions: L and R
        self.observation_space = spaces.Discrete(self.grid_size) # State space: agent position [0 to grid_size-1(4)]

        self.resources = np.zeros(self.grid_size) # Initialization: all 0s
        self.agent_pos = 0 # Agent starts at position 0 

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.agent_pos = np.random.randint(self.grid_size)
        self.resources = np.random.randint(0, 2, size=self.grid_size)
        return self.agent_pos, {}
    
    def step(self, action):
        self.current_step += 1

        if action == 0:
            self.agent_pos = max(0, self.agent_pos - 1)
        else:
            self.agent_pos = min(self.grid_size - 1, self.agent_pos + 1)

        reward = 0
        if self.resources[self.agent_pos] == 1:
            reward = 1
            self.resources[self.agent_pos] = 0 # it picks up the resource
        else:
            reward = -.1

        done = self.current_step >= self.max_steps
        return self.agent_pos, reward, done, False, {}
    
    def render(self):
        grid = ["_"] * self.grid_size
        grid[self.agent_pos] = "A"
        for i, r in enumerate(self.resources):
            if r == 1 and i != self.agent_pos:
                grid[i] = "R"
        print(" ".join(grid))

# TEST TIME:
env = ResourceCollectorEnv()
state, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    env.render()
    print(f"Reward: {reward}")

# Q-Learning:
# Initialize Q-table
Q = np.zeros((env.grid_size, env.action_space.n))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _, _ = env.step(action)
        # Q-learning update
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

print("Trained Q-table:")
print(Q)
