import numpy as np
import random

n_states = 5 # 5 posiciones lineales
n_actions = 2 # 0 = izq, 1 = der
terminal_state = n_states - 1

rewards = np.zeros(n_states)
rewards[terminal_state] = 1 # [0, 0, 0, 0, 1]

Q = np.zeros((n_states, n_actions))

gamma = 0.9 # discou t factor
epsilon = 0.2 # epsilon-greedy policy
episodes = 500

def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.choice([0,1])
    else:
        return np.argmax(Q[state]) 
    
for ep in range(episodes):
    state = 0
    episode = []

    while state != terminal_state:
        action = choose_action(state)
        next_state = state + 1 if action == 1 else max(state - 1, 0)
        reward = rewards[next_state]
        episode.append((state, action, reward))
        state = next_state

    G = 0
    for state, action, reward in reversed(episode):
        G = reward + gamma * G
        Q[state, action] += 0.1 * (G - Q[state, action]) # learning_rate = 0.1

print(f"Q-table final:\n{Q}")

policy = np.argmax(Q, axis=1)
print(f"Política derivada (0=izq, 1=der): {policy}")