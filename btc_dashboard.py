# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

import requests
import joblib
import os
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from predict import fetch_latest_data, prepare_input
from trader import VirtualTrader
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from main import decision_score

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("Bitcoin AI Dashboard")

def is_model_stale(threshold_hours=24):
    try:
        with open("sentiment_history.json", "r") as f:
            data = json.load(f)
            last_time = datetime.fromisoformat(data.get("timestamp"))
            return (datetime.utcnow() - last_time) > timedelta(hours=threshold_hours)
    except Exception as e:
        print(f"[Stale Check Failed]: {e}")
        return True  
    
def retrain_model():
    st.info("Running main.py to retrain model and update sentiment...")
    result = os.system("python main.py")
    if result == 0:
        st.success("Retraining complete.")
    else:
        st.error("Retraining failed.")
        
def get_last_trained_time():
    try:
        with open("sentiment_history.json", "r") as f:
            data = json.load(f)
            ts = data.get("timestamp")
            return datetime.fromisoformat(ts) if ts else None
    except Exception as e:
        print(f"[Last Trained Time Load Error]: {e}")
        return None
    
@st.cache_resource
def load_model_assets():
    model = load_model("btc_best_model.keras")
    scaler = joblib.load("scaler.save")
    columns = joblib.load("feature_columns.save")
    if isinstance(columns, tuple):
        columns = columns[0]
    return model, scaler, list(columns)

model, scaler, feature_columns = load_model_assets()

TRADER_STATE_FILE = "trader_state.json"
BEST_CONFIG_FILE = "best_strategy.pkl"
PREDICTION_LOG_FILE = "prediction_log.json"

if os.path.exists(BEST_CONFIG_FILE):
    best_strategy = joblib.load(BEST_CONFIG_FILE)
    THRESHOLD = best_strategy["Threshold"]
    TAKE_PROFIT = best_strategy["Take Profit"]
    STOP_LOSS = best_strategy["Stop Loss"]
else:
    raise FileNotFoundError("No best_strategy.pkl file found. Please run backtest.py first to generate it.")

selected_section = st.sidebar.radio("Select Section", ["Dashboard", "Trader"])

def get_live_price():
    try:
        res = requests.get("https://api.coingecko.com/api/v3/simple/price", params={
            "ids": "bitcoin",
            "vs_currencies": "usd"
        }, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data['bitcoin']['usd']
    except Exception as e:
        print(f"[get_live_price ERROR] {e}")
        return None
    
def load_best_strategy_config():
    if os.path.exists(BEST_CONFIG_FILE):
        return joblib.load(BEST_CONFIG_FILE)
    else:
        return {
            "Threshold": 0.01,
            "take_profit": 0.0001,
            "stop_loss": -0.0001
        }
        
def save_trader_state(trader, performance):
    def convert(o):
        if isinstance(o, (np.generic, np.bool_)):
            return o.item()
        if isinstance(o, (np.datetime64, pd.Timestamp, datetime)):
            return o.isoformat()
        return str(o)

    with open(TRADER_STATE_FILE, "w") as f:
        json.dump({
            "btc": convert(trader.holdings),
            "cash": convert(trader.balance),
            "history": trader.trade_history,
            "performance": performance,
            "net_worth_history": trader.net_worth_history,
            "cooldown_until": convert(trader.cooldown_until) if trader.cooldown_until else None
        }, f, indent=2, default=convert)

def load_trader_state():
    if os.path.exists(TRADER_STATE_FILE):
        try:
            with open(TRADER_STATE_FILE, "r") as f:
                data = json.load(f)
                vt = VirtualTrader()
                vt.holdings = float(data.get("btc", 0))
                vt.balance = float(data.get("cash", 100000))
                vt.trade_history = data.get("history", [])
                perf = data.get("performance", [])
                vt.net_worth_history = data.get("net_worth_history", [])
                cooldown_str = data.get("cooldown_until")
                vt.cooldown_until = datetime.fromisoformat(cooldown_str) if cooldown_str else None
                return vt, perf
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Load Trader State Error]: {e}")
            print("Resetting trader state due to corrupted file.")
            os.remove(TRADER_STATE_FILE)

    return VirtualTrader(), []

def load_sentiment_signal():
    try:
        with open("sentiment_history.json", "r") as f:
            data = json.load(f)

            last_score = float(data.get("last_score", 0.0))

            signal = (
                "BUY" if last_score > 0.15 else
                "SELL" if last_score < -0.15 else
                "HOLD"
            )

            return {
                "sentiment_score": last_score,
                "signal": signal
            }

    except Exception as e:
        print(f"[Sentiment Load Error]: {e}")
        return {
            "sentiment_score": 0.0,
            "signal": "Unavailable"
        }

def load_prediction_log():
    if os.path.exists(PREDICTION_LOG_FILE):
        with open(PREDICTION_LOG_FILE, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print("prediction_log.json is not a list. Resetting.")
                    return []
            except json.JSONDecodeError:
                print("Corrupted prediction_log.json. Resetting.")
                return []
    return []

def save_prediction_log(log, filename="prediction_log.json"):
    def convert(o):
        if isinstance(o, (np.generic, np.bool_)):
            return o.item()
        if isinstance(o, (np.datetime64, pd.Timestamp)):
            return o.isoformat()
        return str(o)

    with open(filename, "w") as f:
        json.dump(log, f, indent=2, default=convert)
        
if "trader" not in st.session_state or "performance" not in st.session_state:
    st.session_state.trader, st.session_state.performance = load_trader_state()

if selected_section == "Dashboard":
    tab1, tab2 = st.tabs(["Prediction", "Market Overview"])

    with tab1:
        st.markdown("### Model & Sentiment Training/Refresh")
        if st.button("Retrain Model and Backtest Strategy"):
            with st.spinner("Retraining model..."):
                main_result = os.system("python main.py")
                if main_result == 0:
                    st.success("Model retrained successfully!")
                    with st.spinner("Running backtest..."):
                        backtest_result = os.system("python backtest.py")
                        if backtest_result == 0:
                            st.success("Backtest completed successfully!")
                        else:
                            st.error("Backtest failed. Check logs.")
                else:
                    st.error("Model retraining failed. Check logs.")

        with st.spinner("Fetching and preparing data..."):
            df, last_price, sentiment_score = fetch_latest_data()

            if df.isnull().values.any():
                X_input = None
                st.warning("Data contains NaNs. Skipping prediction.")
            else:
                X_input = prepare_input(df, scaler)

        st.metric("Current BTC Price", f"${last_price:,.2f}")

        if X_input is None:
            st.error("Not enough data to make a prediction (need 60 rows).")
        else:
            with st.spinner("Running prediction..."):
                log_return_raw = model.predict(X_input)[0][0]
                sentiment = load_sentiment_signal()

                price_1d_pred = last_price * np.exp(log_return_raw)
                decision_1d = decision_score(last_price, price_1d_pred, sentiment_score)
                price_1d = last_price * (1 + decision_1d)

                predicted_return = np.log(price_1d / last_price)

                st.metric("Predicted Price (Next Day)", f"${price_1d:,.2f}")
                action = "BUY" if price_1d > last_price else "SELL" if price_1d < last_price else "HOLD"
                st.markdown(f"## Suggested Action: **{action}**")

                sentiment = load_sentiment_signal()

                signal_map = {
                    "BUY": "BUY",
                    "SELL": "SELL",
                    "HOLD": "HOLD",
                    "Unavailable": "Unavailable"
                }
                pretty_signal = signal_map.get(sentiment["signal"], "Unavailable")

                st.markdown("### Reddit Sentiment")
                col1, col2 = st.columns(2)
                col1.metric("Score", f"{sentiment['sentiment_score']:.4f}")
                col2.metric("Action", pretty_signal)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index[-60:], df['Close'].tail(60), label='Actual', linewidth=2)
                ax.scatter(df.index[-1], price_1d, color='red', s=60, label='Predicted')
                ax.axhline(y=price_1d, color='red', linestyle='--', alpha=0.4)
                ax.set_title("BTC Close (Last 60 Days)", fontsize=14)
                ax.set_ylabel("Price (USD)")
                ax.set_xlabel("Date")
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend()
                st.pyplot(fig)

    with tab2:
        st.subheader("BTC Daily Returns Overview")
        df, _, _ = fetch_latest_data()
        df['Daily Return (%)'] = df['Close'].pct_change() * 100

        st.markdown("""
        This section helps traders quickly assess how volatile Bitcoin is over time.

        - **Daily Return (%):** Shows the percentage change between each day's closing price.
        - **High spikes** could indicate news or unexpected volatility.
        - Use this to measure **risk exposure** when making or timing trades.
        """)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['Daily Return (%)'], name="Daily % Return", mode='lines'))
        fig3.update_layout(title="Daily BTC Return (%)", xaxis_title="Date", yaxis_title="% Change")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Return Statistics")
        stats = df['Daily Return (%)'].describe()[['mean', 'std', 'min', 'max']]
        stats.index = ['Average Return (%)', 'Volatility (%)', 'Worst Day (%)', 'Best Day (%)']
        st.table(stats.to_frame(name='Daily Return (%)'))

elif selected_section == "Trader":
    st.header("Virtual BTC Trader")

    if 'trader' not in st.session_state or 'performance' not in st.session_state:
        trader, perf = load_trader_state()
        st.session_state.trader = trader
        st.session_state.performance = perf

    if "last_trade_time" not in st.session_state:
        st.session_state.last_trade_time = None

    REFRESH_INTERVAL_SEC = 60
    st_autorefresh(interval=REFRESH_INTERVAL_SEC * 1000, key="refresh_trader")

    live_price = get_live_price()
    df, last_price, sentiment_score = fetch_latest_data()

    if df.isnull().values.any():
        X_input = None
        st.warning("Data contains NaNs. Skipping prediction.")
    else:
        X_input = prepare_input(df, scaler)


    signal = "HOLD"
    now = datetime.utcnow()

    predicted_return = model.predict(X_input)[0][0] if X_input is not None else 0

    price_1d_pred = last_price * np.exp(predicted_return)
    decision = decision_score(last_price, price_1d_pred, sentiment_score)
    adjusted_price = last_price * (1 + decision)

    should_buy = adjusted_price > last_price and st.session_state.trader.balance > 0
    should_sell = adjusted_price < last_price and st.session_state.trader.holdings > 0

    execute_trade = True
    if st.session_state.trader.cooldown_until and datetime.utcnow() < st.session_state.trader.cooldown_until:
        remaining = st.session_state.trader.cooldown_until - datetime.utcnow()
        hours, remainder = divmod(remaining.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        st.warning(f"Cooldown active — next BUY allowed in {hours}h {minutes}m")
        execute_trade = False

    if execute_trade and X_input is not None:
        live_price = get_live_price()

        if live_price is None:
            st.error("Live BTC price unavailable. Cannot proceed with trade evaluation.")
            st.stop()

        predicted_log_return = float(model.predict(X_input)[0][0])
        price_1d_pred = last_price * np.exp(predicted_log_return)
        decision_1d = decision_score(last_price, price_1d_pred, sentiment_score)
        price_1d = last_price * (1 + decision_1d)

        confidence = abs((price_1d - live_price) / live_price)

        min_confidence = 0.05

        if confidence >= min_confidence:
            
            should_buy = adjusted_price > live_price and st.session_state.trader.balance > 0
            should_sell = adjusted_price < live_price and st.session_state.trader.holdings > 0

            if should_buy:
                atr = df["ATR"].iloc[-1] if "ATR" in df.columns else None
                st.session_state.trader.buy(live_price, atr=atr, confidence=confidence)
                st.session_state.last_trade_time = now.isoformat()
                signal = "BUY"
                save_trader_state(st.session_state.trader, st.session_state.performance)
                st.success("BUY executed.")

            elif should_sell:
                st.session_state.trader.sell(live_price)
                st.session_state.last_trade_time = now.isoformat()
                signal = "SELL"
                save_trader_state(st.session_state.trader, st.session_state.performance)
                st.success("SELL executed.")
        else:
            st.write("Confidence too low for trade.")

    if len(df) >= 4:
        actual_return = float(df['Close'].iloc[-1] / df['Close'].iloc[-4] - 1)
        correct = (
            (predicted_return > 0 and actual_return > 0) or
            (predicted_return < 0 and actual_return < 0)
        )
        prediction_log = load_prediction_log()
        prediction_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_change": round(predicted_return * 100, 2),
            "actual_change": round(actual_return * 100, 2),
            "correct": correct
        })
        save_prediction_log(prediction_log)
    else:
        st.warning("Not enough historical data to evaluate prediction accuracy.")

    _, balance, btc = st.session_state.trader.evaluate(
        last_price,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS
    )

    if live_price is None:
        st.error("Live BTC price unavailable. Cannot evaluate trader state.")
        st.stop()
    
    current_price = last_price

    credit = balance + btc * current_price
    avg_price = st.session_state.trader.average_buy_price()

    if btc > 0:
        avg_price = st.session_state.trader.average_buy_price()
        if avg_price is not None and avg_price > 0:
            unrealized_pnl = round((last_price - avg_price) * btc, 2)
        else:
            unrealized_pnl = 0.0
    else:
        unrealized_pnl = 0.0

    st.session_state.performance.append((datetime.utcnow().isoformat(), credit))
    save_trader_state(st.session_state.trader, st.session_state.performance)

    st.metric("Net Worth", f"${credit:,.2f}")
    st.metric("Balance (Cash)", f"${balance:,.2f}")
    st.metric("BTC Held", f"{btc:.6f} BTC")
    st.metric("Unrealized PnL", f"${unrealized_pnl:+,.2f}")

    st.markdown("### Trade History (Total Trades: **{}**)".format(len(st.session_state.trader.trade_history)))

    if st.session_state.trader.trade_history:
        df_trades = pd.DataFrame(st.session_state.trader.trade_history)
        df_trades["Price"] = df_trades["Price"].apply(lambda x: f"${x:,.2f}")
        df_trades["Total"] = df_trades["Total"].apply(lambda x: f"${x:,.2f}")
        df_trades["PnL"] = df_trades.apply(
            lambda row: "-" if row["Type"] == "BUY" or pd.isna(row["PnL"]) else f"${row['PnL']:+,.2f}",
            axis=1
        )
        st.dataframe(df_trades[::-1], use_container_width=True)
        st.metric("Total Trades", st.session_state.trader.total_trades())
    else:
        st.info("No trades made yet.")

    if os.path.exists("best_strategy.pkl"):
        config = joblib.load("best_strategy.pkl")
        st.markdown("### Best Strategy Configuration (Backtest)")
        st.write({
            "Signal Threshold": config['Threshold'],
            "Take Profit": config['Take Profit'],
            "Stop Loss": config['Stop Loss'],
            "Sharpe Ratio": round(config['Sharpe'], 2),
            "Win Rate": f"{config['Win Rate'] * 100:.2f}%"
        })
    else:
        threshold = 0.005
        take_profit = 0.03
        stop_loss = -0.02

    if st.button("Reset Trader"):
        for f in [TRADER_STATE_FILE, PREDICTION_LOG_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.session_state.trader = VirtualTrader()
        st.session_state.performance = []
        st.session_state.last_trade_time = None
        st.rerun()

    st.caption("Trades and predictions run automatically every 60 seconds. ")

