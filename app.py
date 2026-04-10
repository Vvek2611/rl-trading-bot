import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.fetch_data import get_stock_data
from utils.indicators import add_indicators
from env.trading_env import TradingEnv
from stable_baselines3 import DQN
import os

st.set_page_config(
    page_title="RL Trading Bot",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .stMetric { background-color: #0f1525; border-radius: 8px; padding: 10px; }
    h1 { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

st.title("RL Trading Bot")
st.caption("Deep Q-Network agent trained on real stock market data")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Stock", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    initial_cash = st.slider("Starting Capital ($)", 1000, 50000, 10000, 1000)
    st.markdown("---")
    st.markdown("**Model:** DQN via Stable-Baselines3")
    st.markdown("**Indicators:** RSI, MACD, EMA, BB, ATR")
    st.markdown("**Actions:** Buy / Sell / Hold")

@st.cache_data
def load_and_prepare(ticker):
    df = get_stock_data(ticker, period="2y", save=False)
    df = add_indicators(df)
    return df

@st.cache_resource
def load_model():
    if os.path.exists("models/best_model.zip"):
        return DQN.load("models/best_model")
    return None

with st.spinner(f"Fetching {ticker} data..."):
    df = load_and_prepare(ticker)

model = load_model()

split = int(len(df) * 0.8)
test_df = df.iloc[split:].copy().reset_index(drop=True)

if model is None:
    st.error("No trained model found. Please run train.py first.")
    st.stop()

env = TradingEnv(test_df, initial_cash=initial_cash)
obs, _ = env.reset()

portfolio_history = [initial_cash]
actions_taken = []
trade_log = []

with st.spinner("Running bot on test data..."):
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        portfolio_history.append(info['portfolio'])
        actions_taken.append(int(action))
        if done:
            break
    trade_log = env.trade_log if hasattr(env, 'trade_log') else []

start_price = float(test_df.iloc[0]['Close'])
bh = [initial_cash * float(test_df.iloc[i]['Close']) / start_price
      for i in range(len(test_df))]

final_pv   = portfolio_history[-1]
total_ret  = (final_pv - initial_cash) / initial_cash * 100
bh_ret     = (bh[-1] - initial_cash) / initial_cash * 100
alpha      = total_ret - bh_ret
n_buys     = actions_taken.count(1)
n_sells    = actions_taken.count(2)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Starting Capital", f"${initial_cash:,.0f}")
col2.metric("Final Portfolio",  f"${final_pv:,.0f}",
            f"{total_ret:+.2f}%")
col3.metric("DQN Return",       f"{total_ret:+.2f}%")
col4.metric("Buy & Hold",       f"{bh_ret:+.2f}%")
col5.metric("Alpha",            f"{alpha:+.2f}%",
            "outperformed" if alpha > 0 else "underperformed")

st.markdown("---")

fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.65, 0.35],
                    subplot_titles=("Portfolio Value", "AAPL Price + Signals"))

steps = list(range(len(portfolio_history)))

fig.add_trace(go.Scatter(
    x=steps, y=portfolio_history,
    name="DQN Agent",
    line=dict(color="#3b82f6", width=2)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=list(range(len(bh))), y=bh,
    name="Buy & Hold",
    line=dict(color="#f59e0b", width=1.5, dash="dash")
), row=1, col=1)

fig.add_hline(y=initial_cash, line_dash="dot",
              line_color="gray", row=1, col=1)

prices = test_df['Close'].values
fig.add_trace(go.Scatter(
    x=list(range(len(prices))), y=prices,
    name="Price", line=dict(color="#94a3b8", width=1.5)
), row=2, col=1)

buy_steps  = [t['step'] - env.window if hasattr(env,'window') else t['step']
              for t in trade_log if t['action'] == 'BUY']
sell_steps = [t['step'] - env.window if hasattr(env,'window') else t['step']
              for t in trade_log if t['action'] == 'SELL']
buy_prices  = [t['price'] for t in trade_log if t['action'] == 'BUY']
sell_prices = [t['price'] for t in trade_log if t['action'] == 'SELL']

fig.add_trace(go.Scatter(
    x=buy_steps, y=buy_prices,
    mode="markers", name="Buy",
    marker=dict(symbol="triangle-up", size=12, color="#22c55e")
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=sell_steps, y=sell_prices,
    mode="markers", name="Sell",
    marker=dict(symbol="triangle-down", size=12, color="#ef4444")
), row=2, col=1)

fig.update_layout(
    height=600,
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0f1525",
    font=dict(color="#c9d1e0"),
    legend=dict(bgcolor="#0f1525", bordercolor="#1e2d4a"),
    margin=dict(l=0, r=0, t=30, b=0)
)
fig.update_xaxes(gridcolor="#1e2d4a")
fig.update_yaxes(gridcolor="#1e2d4a")

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trade Summary")
    summary = pd.DataFrame({
        "Metric": ["Total Trades", "Buy Orders",
                   "Sell Orders", "Hold Actions"],
        "Value": [n_buys + n_sells, n_buys,
                  n_sells, actions_taken.count(0)]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

with col2:
    st.subheader("Performance")
    perf = pd.DataFrame({
        "Metric": ["DQN Return", "Buy & Hold",
                   "Alpha", "Final Value"],
        "Value": [f"{total_ret:+.2f}%", f"{bh_ret:+.2f}%",
                  f"{alpha:+.2f}%", f"${final_pv:,.2f}"]
    })
    st.dataframe(perf, hide_index=True, use_container_width=True)

st.caption("Built with Stable-Baselines3 · PyTorch · Streamlit · Vvek2611")