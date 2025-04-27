import numpy as np
import yfinance as yf
import ta
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
from sentiment import get_daily_sentiment
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32

FEATURE_COLUMNS = [
    'RSI', 'MACD', 'EMA_50', 'EMA_200', 'BB_Percent_B',
    'BB_Upper', 'BB_Lower', 'Volume', 'Momentum', 'Stoch_%K', 'CCI',
    'Williams_%R', 'ATR', 'Log_Return', 'Price_EMA_Ratio', 'Volatility',
    'Return_Residual'
]

class CustomModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_score = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if loss is not None and val_loss is not None:
            combo_score = (loss + val_loss) + abs(loss - val_loss)
            if combo_score < self.best_score:
                self.best_score = combo_score
                self.model.save(self.filepath)
                print(f"\nSaved best model at epoch {epoch + 1} with score: {self.best_score:.5f}")

def fetch_data():
    df = yf.download('BTC-USD', start='2017-01-01', end='2024-01-01')
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    df['MACD'] = ta.trend.MACD(close).macd()

    ema_50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
    ema_200 = ta.trend.EMAIndicator(close, 200).ema_indicator()
    df['EMA_50'] = ema_50
    df['EMA_200'] = ema_200

    boll = ta.volatility.BollingerBands(close)
    df['BB_Upper'] = boll.bollinger_hband()
    df['BB_Lower'] = boll.bollinger_lband()
    df['BB_Percent_B'] = boll.bollinger_pband()

    df['Momentum'] = ta.momentum.ROCIndicator(close).roc()
    df['Stoch_%K'] = ta.momentum.StochasticOscillator(close, high, low).stoch()
    df['CCI'] = ta.trend.CCIIndicator(close, high, low).cci()
    df['Williams_%R'] = ta.momentum.WilliamsRIndicator(close, high, low).williams_r()
    df['ATR'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df['Return_Residual'] = close - ema_50

    df['Log_Return'] = np.log(close / close.shift(1)).shift(1)
    df['Price_EMA_Ratio'] = (close / ema_50).shift(1)
    df['Volatility'] = np.log(close / close.shift(1)).rolling(window=10).std().shift(1)

    df['Target'] = close.shift(-1)
    df['Log_Target'] = np.log(df['Target'] / close)

    df.dropna(inplace=True)
    return df[FEATURE_COLUMNS], df['Log_Target'], close, df['ATR'], df['Momentum'], df['Volume']

def preprocess_data(features, price_target):
    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(features_scaled)):
        X.append(features_scaled[i - SEQUENCE_LENGTH:i])
        y.append(price_target.values[i])
    return np.array(X), np.array(y), feature_scaler

def build_price_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(128)(inp)
    x = Dropout(0.3)(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model

def run():
    features, price_target, closes, atr, momentum, volume = fetch_data()

    X, y, feat_scaler = preprocess_data(features, price_target)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    close_test = closes.iloc[-len(y_test):].values

    model = build_price_model((X.shape[1], X.shape[2]))
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-5),
            CustomModelSaver("btc_best_model.keras")
        ],
        verbose=1
    )

    pred_scaled = model.predict(X_test).flatten()
    close_base = close_test[-len(pred_scaled):]
    pred = close_base * np.exp(pred_scaled)
    y_true = close_base * np.exp(y_test)
    
    sentiment_score, _ = get_daily_sentiment()
    joblib.dump(sentiment_score, "latest_sentiment_score.save")

    joblib.dump(FEATURE_COLUMNS, "feature_columns.save")
    joblib.dump(feat_scaler, "scaler.save")

    predicted_price = close_test[-1] * np.exp(pred_scaled[-1])
    price_now = close_test[-1]

    decision = decision_score(price_now, predicted_price, sentiment_score)
    print(f"\nCombined Decision Score: {decision:.4f}")

    print("\nEvaluation Metrics:")
    print(f"MAE: {mean_absolute_error(y_true, pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, pred)*100:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True Future Price", linewidth=2)
    plt.plot(pred, label="Predicted Future Price", linestyle="--", linewidth=2)
    plt.title("Bitcoin Price Prediction vs Actual", fontsize=16)
    plt.xlabel("Time (days)", fontsize=14)
    plt.ylabel("Price (USD)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png", dpi=300)
    plt.show()

def decision_score(price_now, predicted_price, reddit_sent):
    base_weights = {
        "prediction": 0.85,
        "reddit": 0.15,
    }

    try:
        pred_score = (predicted_price - price_now) / price_now
        pred_score = max(min(pred_score * 1.5, 1), -1)
    except:
        base_weights["prediction"] = 0
        pred_score = 0

    sentiment_adj = float(reddit_sent)
    if reddit_sent == 0:
        base_weights["reddit"] = 0

    total_weight = sum(base_weights.values())
    weights = {k: v / total_weight for k, v in base_weights.items()}

    decision = (
        pred_score * weights["prediction"] +
        sentiment_adj * weights["reddit"]
    )
    
    return decision

if __name__ == '__main__':
    run()