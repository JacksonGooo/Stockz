#!/usr/bin/env python3
"""
Test live BTC prediction with real-time data
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES

def fetch_live_candles(symbol='BTCUSDT', limit=100):
    """Fetch live 1-minute candles from Binance"""
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': '1m',
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    candles = []
    for kline in data:
        candles.append({
            'timestamp': kline[0],
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5])
        })

    return pd.DataFrame(candles)


def main():
    print("=" * 60)
    print("LIVE BTC PREDICTION TEST")
    print("=" * 60)

    # Try to load the autoregressive model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'autoregressive_btc', 'model.keras')

    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Looking for alternative models...")

        # Check for other models
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(models_dir):
            for d in os.listdir(models_dir):
                m_path = os.path.join(models_dir, d, 'model.keras')
                if os.path.exists(m_path):
                    print(f"  Found: {m_path}")
        return

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    print()

    # Fetch live candles
    print("Fetching live BTC candles from Binance...")
    try:
        df = fetch_live_candles(limit=100)
        print(f"Fetched {len(df)} candles")
    except Exception as e:
        print(f"Error fetching candles: {e}")
        # Try loading from local data
        print("Using local data instead...")
        from data_loader import load_all_btc_candles
        all_candles = load_all_btc_candles()
        df = all_candles.tail(100).reset_index(drop=True)

    current_price = df['close'].iloc[-1]
    current_time = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')

    print(f"\nCurrent Time: {current_time}")
    print(f"Current Price: ${current_price:,.2f}")
    print()

    # Calculate features
    print("Calculating features...")
    features = calculate_all_features(df)
    normalized = rolling_zscore(features, window=60)
    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)
    feature_matrix = normalized[FEATURE_NAMES].values.astype(np.float32)

    # Take last 30 for model input
    sequence_length = 30
    input_seq = feature_matrix[-sequence_length:]

    # Predict
    print("Making 60-minute prediction...")
    model_input = input_seq.reshape(1, sequence_length, NUM_FEATURES)
    predictions = model.predict(model_input, verbose=0)

    # The model outputs shape (1, 60, 5) for 60 steps of OHLCV returns
    pred_returns = predictions[0]  # Shape (60, 5)

    print(f"\nPrediction output shape: {pred_returns.shape}")
    print(f"Close return predictions (first 10):")
    for i in range(min(10, len(pred_returns))):
        print(f"  +{i+1} min: {pred_returns[i, 3]*100:.4f}%")

    # Cumulative close return
    close_returns = pred_returns[:, 3]  # Close is index 3
    cumulative_return = np.sum(close_returns)
    predicted_price = current_price * (1 + cumulative_return)

    print(f"\n--- 60-MINUTE FORECAST ---")
    print(f"Sum of predicted returns: {cumulative_return*100:.4f}%")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Price Change: ${predicted_price - current_price:+,.2f} ({cumulative_return*100:+.3f}%)")

    # Direction
    direction = "UP" if cumulative_return > 0 else "DOWN"
    print(f"\nDirection: {direction}")

    # Trajectory
    print("\n--- PRICE TRAJECTORY ---")
    prices = [current_price]
    running_price = current_price
    for i, ret in enumerate(close_returns):
        running_price = running_price * (1 + ret)
        if (i + 1) % 10 == 0:
            print(f"  +{i+1:2d} min: ${running_price:,.2f} ({(running_price/current_price - 1)*100:+.3f}%)")
        prices.append(running_price)

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
