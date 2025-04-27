# -*- coding: utf-8 -*-
import numpy as np
import yfinance as yf
import ta
import joblib
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from main import decision_score

SEQUENCE_LENGTH = 60

FEATURE_COLUMNS = joblib.load("feature_columns.save")
scaler = joblib.load("scaler.save")
regression_model = load_model("btc_best_model.keras")

def fetch_latest_data():
    df = yf.download('BTC-USD', period='370d', interval='1d', auto_adjust=False)
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]
    if df.empty:
        raise ValueError("Could not fetch BTC data.")

    close = df['Close']
    df['Close'] = close

    try:
        df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        macd = ta.trend.MACD(close=close)
        df['MACD'] = macd.macd()
        df['EMA_50'] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        df['Return_Residual'] = df['Close'] - df['EMA_50']
        df['EMA_200'] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()
        boll = ta.volatility.BollingerBands(close=close)
        df['BB_Upper'] = boll.bollinger_hband()
        df['BB_Lower'] = boll.bollinger_lband()
        df['BB_Percent_B'] = boll.bollinger_pband()
        df['Momentum'] = ta.momentum.ROCIndicator(close=close).roc()
        df['Stoch_%K'] = ta.momentum.StochasticOscillator(close, df['High'], df['Low']).stoch()
        df['CCI'] = ta.trend.CCIIndicator(close, df['High'], df['Low']).cci()
        df['Williams_%R'] = ta.momentum.WilliamsRIndicator(close, df['High'], df['Low']).williams_r()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], close).average_true_range()
        df['Log_Return'] = np.log(close / close.shift(1))
        df['Price_EMA_Ratio'] = close / df['EMA_50']
        df['Volatility'] = df['Log_Return'].rolling(window=10).std()
    except Exception as e:
        raise RuntimeError(f"Error while computing indicators: {e}")

    sentiment_score = joblib.load("latest_sentiment_score.save")
    if "Sentiment" in FEATURE_COLUMNS:
        df["Sentiment"] = sentiment_score

    df.dropna(inplace=True)

    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame after dropna: {missing_cols}")

    return df, close.iloc[-1], sentiment_score

def prepare_input(df, scaler):
    df = df.copy()
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    df = df[FEATURE_COLUMNS]
    if df.isnull().any().any():
        raise ValueError("NaNs found in DataFrame before scaling.")

    scaled = scaler.transform(df)
    last_seq = scaled[-SEQUENCE_LENGTH:]
    return np.expand_dims(last_seq, axis=0)

def get_signal(score, threshold=0.15):
        if score > threshold:
            return "UP"
        elif score < -threshold:
            return "DOWN"
        else:
            return "HOLD"

def predict():
    df, last_close_price, sentiment_score = fetch_latest_data()

    X_input = prepare_input(df, scaler)

    log_return_pred = regression_model.predict(X_input)[0][0]
    
    sentiment_score = joblib.load("latest_sentiment_score.save")

    price_1d_pred = last_close_price * np.exp(log_return_pred)
    decision_1d = decision_score(last_close_price, price_1d_pred, sentiment_score)
    price_1d = last_close_price * (1 + decision_1d)

    delta_1d = price_1d - last_close_price
    if price_1d > last_close_price:
        action = "BUY"
    elif price_1d < last_close_price:
        action = "SELL"
    else:
        action = "HOLD"

    print(f"\nLast BTC Close: ${last_close_price:,.2f}")
    print(f"Next Day Price:  ${price_1d:,.2f}")
    print(f"Predicted Log Return (1D): {log_return_pred:.5f}")
    print(f"Change (1D): {delta_1d:+.2f}")
    print(f"Suggested Action: {action}")

    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "last_close_price": float(round(last_close_price, 2)),
        "decision_score_1d": float(round(decision_1d, 4)),
        "log_return": float(round(log_return_pred, 5)),
        "price_delta_1d": float(round(delta_1d, 2)),
        "action": str(action),
        "sentiment": float(round(sentiment_score, 4)),
        "decision_score": float(round(decision_1d, 4))
    }

    with open("prediction_log.json", "a") as f:
        f.write(json.dumps(log) + "\n")

if __name__ == '__main__':
    predict()