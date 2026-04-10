import gymnasium as gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, initial_cash=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.df.columns = [col[0] if isinstance(col, tuple) else col for col in self.df.columns]
        self.initial_cash = initial_cash
        self.feature_cols = [
            'Close', 'EMA_20', 'EMA_50', 'RSI',
            'MACD', 'MACD_Signal', 'BB_Width',
            'ATR', 'Volume_Ratio', 'MACD_Hist'
        ]
        self.window = 10
        n_features = len(self.feature_cols) * self.window + 3
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )

    def _get_obs(self):
        w = self.df.iloc[self.idx - self.window: self.idx]
        features = []
        for col in self.feature_cols:
            arr = w[col].values.astype(float)
            features.extend((arr - arr.mean()) / (arr.std() + 1e-8))
        pv = self.cash + self.shares * float(self.df.iloc[self.idx]['Close'])
        features += [
            float(self.position),
            (pv - self.initial_cash) / self.initial_cash,
            self.cash / self.initial_cash
        ]
        return np.array(features, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx       = self.window
        self.cash      = float(self.initial_cash)
        self.shares    = 0
        self.position  = 0
        self.buy_price = 0.0
        self.portfolio_history = [self.initial_cash]
        return self._get_obs(), {}

    def step(self, action):
        price  = float(self.df.iloc[self.idx]['Close'])
        reward = 0.0

        if action == 1 and self.position == 0:
            self.shares    = int(self.cash * 0.95 / price)
            self.cash     -= self.shares * price * 1.001
            self.buy_price = price
            self.position  = 1
            reward         = -0.1

        elif action == 2 and self.position == 1:
            self.cash    += self.shares * price * 0.999
            reward        = (price - self.buy_price) / self.buy_price * 100
            self.shares   = 0
            self.position = 0

        elif action == 0 and self.position == 1:
            if self.idx + 1 < len(self.df):
                next_price = float(self.df.iloc[self.idx + 1]['Close'])
                reward     = (next_price - price) / price * 10

        pv = self.cash + self.shares * price
        self.portfolio_history.append(pv)
        self.idx += 1
        done = self.idx >= len(self.df) - 1

        return self._get_obs(), reward, done, False, {'portfolio': pv}