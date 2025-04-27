import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr

SEQUENCE_LENGTH = 60 

def fetch_data():
    df = yf.download('BTC-USD', start='2024-01-01', interval='1d')

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
    return df


def prepare_sequences(df, feature_columns, scaler, seq_length=60):
    features = df[feature_columns]
    features_scaled = scaler.transform(features)

    X = []
    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i - seq_length:i])
    return np.array(X)

def backtest_predictions(close_prices, predicted_prices, actual_prices, momentum,
                         threshold=0.005, stop_loss=-0.02, take_profit=0.05):

    close_prices = np.asarray(close_prices).flatten()
    predicted_prices = np.asarray(predicted_prices).flatten()
    actual_prices = np.asarray(actual_prices).flatten()
    momentum = np.asarray(momentum).flatten()

    min_len = min(len(close_prices), len(predicted_prices), len(actual_prices), len(momentum))
    close_prices = close_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]
    actual_prices = actual_prices[:min_len]
    momentum = momentum[:min_len]

    signals = []
    positions = []
    portfolio_returns = []
    trade_returns = []
    in_position = False
    entry_price = 0
    position_direction = 0

    for i in range(1, min_len):
        current_price = close_prices[i]
        prev_price = close_prices[i - 1]
        predicted_delta = predicted_prices[i] - predicted_prices[i - 1]

        if in_position:
            current_return = (current_price - entry_price) / entry_price * position_direction
            if current_return <= stop_loss or current_return >= take_profit:
                signals.append('EXIT')
                portfolio_returns.append(current_return)
                trade_returns.append(current_return)
                positions.append(0)
                in_position = False
                continue

        if not in_position:
            if predicted_delta > threshold:
                signals.append('BUY')
                positions.append(1)
                entry_price = current_price
                position_direction = 1
                in_position = True
            elif predicted_delta < -threshold:
                signals.append('SELL')
                positions.append(-1)
                entry_price = current_price
                position_direction = -1
                in_position = True
            else:
                signals.append('HOLD')
                positions.append(0)
        else:
            signals.append('HOLD')
            positions.append(position_direction)

        if in_position:
            daily_return = (current_price - prev_price) / prev_price * position_direction
            portfolio_returns.append(daily_return)
        else:
            portfolio_returns.append(0)

    if in_position:
        final_return = (close_prices[-1] - entry_price) / entry_price * position_direction
        signals.append('EXIT')
        portfolio_returns.append(final_return)
        trade_returns.append(final_return)
        positions.append(0)

    while len(signals) < len(close_prices):
        signals.append('HOLD')
        positions.append(0)
        portfolio_returns.append(0.0)

    results = pd.DataFrame({
        "Date": pd.date_range(start='2020-01-01', periods=len(close_prices)),
        "Close": close_prices,
        "Predicted": predicted_prices,
        "Actual": actual_prices,
        "Signal": signals,
        "Position": positions,
        "Daily_Return": portfolio_returns,
        "Cumulative_Return": np.cumsum(portfolio_returns)
    })

    return results

def evaluate_backtest(results):
    if results.empty or len(results) < 10:
        print("Not enough data to evaluate.")
        return None

    trades = []
    entry_idx = None
    for i, row in results.iterrows():
        if row['Signal'] in ['BUY', 'SELL']:
            entry_idx = i
        elif row['Signal'] == 'EXIT' and entry_idx is not None:
            trade_return = results.loc[entry_idx:i, 'Daily_Return'].sum()
            start_date = results.loc[entry_idx, 'Date']
            end_date = results.loc[i, 'Date']
            trade_duration = (end_date - start_date).days
            trades.append((trade_return, trade_duration))
            entry_idx = None

    trade_returns = np.array([t[0] for t in trades])
    durations = np.array([t[1] for t in trades])

    total_return = results['Cumulative_Return'].iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(results)) - 1
    returns = results['Daily_Return']

    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 1e-6 else 0.0
    win_rate = (trade_returns > 0).mean()
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    max_drawdown = (returns.cumsum().cummax() - returns.cumsum()).max()

    print("\nBacktest Evaluation Results:")
    print(f"Total Return: {total_return:.4f}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trade_returns)
    }

def run_backtest():
    print("\n=== Starting BTC Model Backtest ===")

    model = load_model("btc_best_model.keras")
    scaler = joblib.load("scaler.save")
    feature_columns = joblib.load("feature_columns.save")

    df = fetch_data()

    X = prepare_sequences(df, feature_columns, scaler)
    preds = model.predict(X).flatten()

    closes = df['Close'].values[SEQUENCE_LENGTH:]
    log_targets = df['Log_Target'].values[SEQUENCE_LENGTH:]
    momentum = df['Momentum'].values[SEQUENCE_LENGTH:]
    dates = df.index.values[SEQUENCE_LENGTH:]

    min_len = min(len(closes), len(preds), len(log_targets), len(momentum))
    closes = closes[:min_len]
    preds = preds[:min_len]
    log_targets = log_targets[:min_len]
    momentum = momentum[:min_len]
    dates = dates[:min_len]

    true_prices = closes * np.exp(log_targets)
    pred_prices = closes * np.exp(preds)

    thresholds = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    stop_losses = [-0.005, -0.01, -0.015, -0.02]
    take_profits = [0.02, 0.03, 0.04, 0.05, 0.07, 0.1]

    best_result = None
    best_config = None

    print("\n=== Searching for best parameters based on Win Rate ===")
    for threshold in thresholds:
        for stop_loss in stop_losses:
            for take_profit in take_profits:
                results = backtest_predictions(
                    closes, pred_prices, true_prices, momentum,
                    threshold=threshold,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                metrics = evaluate_backtest(results)
                if metrics:
                    print(f"Tested: Threshold={threshold}, SL={stop_loss}, TP={take_profit} → Win Rate={metrics['win_rate']:.2%}")
                    if best_result is None or metrics['win_rate'] > best_result['win_rate']:
                        best_result = metrics
                        best_config = {
                            "Threshold": threshold,
                            "Stop Loss": stop_loss,
                            "Take Profit": take_profit,
                            "Win Rate": metrics['win_rate'],
                            "Sharpe": metrics['sharpe_ratio'],
                            "Profit Factor": metrics['profit_factor'],
                            "Total Return": metrics['total_return']
                        }

    print("\nLSTM Model Evaluation Metrics (Backtest):")
    correlation = pearsonr(true_prices.ravel(), pred_prices.ravel())[0]
    print(f"MAE: {mean_absolute_error(true_prices, pred_prices):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(true_prices, pred_prices)):.4f}")
    print(f"R² Score: {r2_score(true_prices, pred_prices):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(true_prices, pred_prices) * 100:.2f}%")
    print(f"Correlation: {correlation:.2f}")

    if best_result:
        print("\nBest Strategy Found:")
        for k, v in best_config.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        joblib.dump(best_config, "best_strategy.pkl")
        print("\nBest strategy saved to best_strategy.pkl")
    else:
        print("No successful backtest found.")

if __name__ == "__main__":
    run_backtest()