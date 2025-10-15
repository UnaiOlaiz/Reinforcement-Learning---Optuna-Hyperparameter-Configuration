import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

data = pd.read_csv("/home/unai-olaizola-osa/Documents/RL/ETXEA/Google_Stock_Price_Test.xls", index_col=0)
prices = data["Close"].values

class StockTradingEnv(gym.Env):
    def __init__(self, prices, window=10, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.prices = prices
        self.window = window
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3) # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window, ), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.window
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        return self.prices[self.current_step - self.window:self.current_step]
    
    def step(self, action):
        current_price = self.prices[self.current_step]

        if action == 1: # buy
            if self.balance >= self.current_price:
                self.shares += 1
                self.balance -= current_price
        elif action == 2: # sell
            if self.shares > 0:
                self.shares -= 1
                self.balance += current_price
        
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            self.done = True

        portfolio_value = self.balance + self.shares * current_price
        reward = portfolio_value - self.initial_balance
        state = self._get_state()

        return state, reward, self.done, {}
    
env = StockTradingEnv(prices)
state = env.reset()
print("Initial state:", state)