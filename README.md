# RL Trading Bot

A real-world stock trading bot using Deep Q-Network (DQN)
trained on live AAPL market data via Yahoo Finance.

## Results
| Metric | Value |
|--------|-------|
| Starting Capital | $10,000 |
| Final Portfolio | $11,478 |
| DQN Return | +14.79% |
| Buy & Hold Return | +9.20% |
| Alpha (outperformance) | +5.59% |

## Features
- Live data from Yahoo Finance via yfinance
- 8 technical indicators — RSI, MACD, EMA, Bollinger Bands, ATR
- Custom OpenAI Gymnasium trading environment
- DQN agent via Stable-Baselines3 and PyTorch
- Backtest vs Buy & Hold benchmark

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python train.py      # Train the agent
python evaluate.py   # Backtest results
```

## Tech Stack
Python · Gymnasium · Stable-Baselines3 · PyTorch · yfinance · ta
