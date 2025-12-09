#!/usr/bin/env python3
"""
Web API Prediction Script for Autoregressive BTC Model
Accepts candles via stdin, returns predictions as JSON
"""

import os
import sys
import json
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'autoregressive_btc')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')

# Prediction settings - strict caps due to error compounding
# Backtest showed 50% accuracy when chaining predictions - heavily constrain outputs
MAX_SINGLE_CANDLE_RETURN = 0.001  # 0.1% max per candle (typical BTC 1-min move)
MAX_CUMULATIVE_RETURN = 0.02     # 2% max over 60 minutes
DECAY_FACTOR = 0.95              # Decay predictions over time


def load_model():
    """Load the trained autoregressive model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

    return model, config


def prepare_features(candles: list) -> np.ndarray:
    """Convert candles to feature matrix"""
    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Ensure required columns
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate features
    features = calculate_all_features(df)

    # Apply rolling z-score normalization
    normalized = rolling_zscore(features, window=60)

    # Fill NaN values
    normalized = normalized.fillna(0)

    # Convert to numpy array with correct feature order
    feature_matrix = normalized[FEATURE_NAMES].values

    return feature_matrix


def predict_autoregressive(model, features: np.ndarray, num_predictions: int = 60) -> list:
    """
    Generate predictions autoregressively
    Each prediction is fed back as input for the next
    """
    sequence_length = 30  # Model expects 30 timesteps

    if len(features) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} candles, got {len(features)}")

    # Take last sequence_length features
    current_sequence = features[-sequence_length:].copy()

    predictions = []
    cumulative_return = 0

    for i in range(num_predictions):
        # Reshape for model: (1, sequence_length, num_features)
        model_input = current_sequence.reshape(1, sequence_length, NUM_FEATURES)

        # Predict next candle returns [open, high, low, close, volume]
        pred_returns = model.predict(model_input, verbose=0)[0]

        # Apply decay - predictions become less reliable over time
        decay = DECAY_FACTOR ** i
        pred_returns = pred_returns * decay

        # Cap individual returns to realistic values
        pred_returns = np.clip(pred_returns, -MAX_SINGLE_CANDLE_RETURN, MAX_SINGLE_CANDLE_RETURN)

        # Check cumulative return limit
        close_return = pred_returns[3]  # close_return is index 3
        if abs(cumulative_return + close_return) > MAX_CUMULATIVE_RETURN:
            # Scale down to stay within limit
            scale = (MAX_CUMULATIVE_RETURN - abs(cumulative_return)) / abs(close_return) if close_return != 0 else 0
            pred_returns = pred_returns * scale
            close_return = pred_returns[3]

        cumulative_return += close_return

        predictions.append({
            'open_return': float(pred_returns[0]),
            'high_return': float(pred_returns[1]),
            'low_return': float(pred_returns[2]),
            'close_return': float(pred_returns[3]),
            'volume_change': float(pred_returns[4])
        })

        # Create new feature row for next prediction
        # Use predicted returns as the OHLCV features, keep other features similar
        new_features = current_sequence[-1].copy()
        new_features[0] = pred_returns[0]  # open_return
        new_features[1] = pred_returns[1]  # high_return
        new_features[2] = pred_returns[2]  # low_return
        new_features[3] = pred_returns[3]  # close_return
        new_features[4] = pred_returns[4]  # volume_change

        # Slide window: remove oldest, add new
        current_sequence = np.vstack([current_sequence[1:], new_features])

    return predictions


def returns_to_candles(predictions: list, last_candle: dict) -> list:
    """Convert predicted returns to actual OHLCV candles"""
    candles = []
    current_close = last_candle['close']
    current_volume = last_candle['volume']
    last_timestamp = last_candle['timestamp']

    for i, pred in enumerate(predictions):
        # Calculate prices from returns
        new_open = current_close * (1 + pred['open_return'])
        new_high = current_close * (1 + pred['high_return'])
        new_low = current_close * (1 + pred['low_return'])
        new_close = current_close * (1 + pred['close_return'])
        new_volume = current_volume * (1 + pred['volume_change'])

        # Ensure high >= max(open, close) and low <= min(open, close)
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)

        # Ensure volume is positive
        new_volume = max(new_volume, 0)

        candles.append({
            'timestamp': last_timestamp + (i + 1) * 60000,  # +1 minute each
            'open': float(new_open),
            'high': float(new_high),
            'low': float(new_low),
            'close': float(new_close),
            'volume': float(new_volume)
        })

        current_close = new_close
        current_volume = new_volume

    return candles


def main():
    """Main entry point for API calls"""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        candles = input_data.get('candles', [])
        output_minutes = input_data.get('outputMinutes', 60)

        if len(candles) < 60:
            print(json.dumps({
                'error': 'Not enough data',
                'message': f'Need at least 60 candles for feature calculation, got {len(candles)}'
            }))
            sys.exit(1)

        # Load model
        model, config = load_model()

        # Prepare features
        features = prepare_features(candles)

        # Generate predictions
        pred_returns = predict_autoregressive(model, features, output_minutes)

        # Convert to candles
        last_candle = candles[-1]
        pred_candles = returns_to_candles(pred_returns, last_candle)

        # Calculate summary
        current_price = last_candle['close']
        final_price = pred_candles[-1]['close']
        percent_change = ((final_price - current_price) / current_price) * 100

        # Determine direction
        if percent_change > 0.1:
            direction = 'up'
        elif percent_change < -0.1:
            direction = 'down'
        else:
            direction = 'neutral'

        result = {
            'predictions': pred_candles,
            'summary': {
                'currentPrice': current_price,
                'predictedPrice': final_price,
                'percentChange': percent_change,
                'direction': direction
            },
            'model': {
                'type': 'autoregressive_lstm',
                'directionAccuracy': 0.50,  # Backtest showed 50% accuracy with chained predictions
                'sequenceLength': 30,
                'predictionHorizon': output_minutes,
                'note': 'Speculative - autoregressive predictions have compounding error'
            }
        }

        print(json.dumps(result))

    except FileNotFoundError as e:
        print(json.dumps({'error': 'Model not found', 'message': str(e)}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': 'Prediction failed', 'message': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
