from data.fetch_data import get_stock_data
from env.trading_env import TradingEnv
from utils.indicators import add_indicators
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

TICKER    = "AAPL"
TIMESTEPS = 200_000

df       = get_stock_data(TICKER, period="4y")
df       = add_indicators(df)
split    = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()

train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
eval_env  = DummyVecEnv([lambda: TradingEnv(test_df)])

os.makedirs("models", exist_ok=True)
model = DQN("MlpPolicy", train_env,
            learning_rate=1e-4, buffer_size=50_000,
            learning_starts=1_000, batch_size=64,
            gamma=0.99, verbose=1,
            tensorboard_log="./tensorboard_logs/")

eval_cb = EvalCallback(eval_env,
                       best_model_save_path="./models/",
                       eval_freq=5_000, verbose=1)

print("Training started...")
model.learn(total_timesteps=TIMESTEPS, callback=eval_cb)
model.save("models/dqn_final")
print("Training done! Model saved.")