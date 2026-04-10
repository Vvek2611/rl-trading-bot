from data.fetch_data import load_stock_data
from env.trading_env import TradingEnv
from utils.indicators import add_indicators
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os

TICKER  = "AAPL"
df      = load_stock_data(TICKER)
df      = add_indicators(df)
split   = int(len(df) * 0.8)
test_df = df.iloc[split:].copy().reset_index(drop=True)

env     = TradingEnv(test_df)
model   = DQN.load("models/best_model")
obs, _  = env.reset()
history = [10000]

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, info = env.step(int(action))
    history.append(info['portfolio'])
    if done:
        break

start = float(test_df.iloc[0]['Close'])
bh    = [10000 * float(test_df.iloc[i]['Close']) / start for i in range(len(test_df))]

print(f"Final Portfolio : ${history[-1]:,.2f}")
print(f"Return          : {(history[-1]-10000)/10000*100:+.2f}%")
print(f"Buy & Hold      : {(bh[-1]-10000)/10000*100:+.2f}%")

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(12,5))
plt.plot(history, label="DQN Agent", color="blue")
plt.plot(bh,      label="Buy & Hold", color="orange", linestyle="--")
plt.title(f"DQN Agent vs Buy & Hold — {TICKER}")
plt.xlabel("Step"); plt.ylabel("Portfolio ($)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("results/backtest.png")
plt.show()
print("Chart saved to results/backtest.png")