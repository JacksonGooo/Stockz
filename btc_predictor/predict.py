#!/usr/bin/env python3
"""
LSTM Model Prediction Script
Loads a trained Keras model and predicts future candles
"""

import os
import sys
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def load_model_and_stats(category: str, asset: str):
    """Load the trained model and normalization stats"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', f'{asset.lower()}_lstm.keras')
    stats_path = os.path.join(base_dir, 'models', f'{asset.lower()}_lstm_stats.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    model = tf.keras.models.load_model(model_path)
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return model, stats

def normalize_candles(candles: list, stats: dict) -> np.ndarray:
    """Normalize candle data using saved min/max stats"""
    min_price = stats['min_price']
    max_price = stats['max_price']
    min_volume = stats['min_volume']
    max_volume = stats['max_volume']

    price_range = max_price - min_price
    volume_range = max_volume - min_volume

    normalized = []
    for candle in candles:
        norm_candle = [
            (candle['open'] - min_price) / price_range,
            (candle['high'] - min_price) / price_range,
            (candle['low'] - min_price) / price_range,
            (candle['close'] - min_price) / price_range,
            (candle['volume'] - min_volume) / volume_range if volume_range > 0 else 0
        ]
        normalized.append(norm_candle)

    return np.array(normalized)

def denormalize_price(value: float, stats: dict) -> float:
    """Convert normalized price back to actual price"""
    min_price = stats['min_price']
    max_price = stats['max_price']
    return value * (max_price - min_price) + min_price

def predict_future(model, input_sequence: np.ndarray, stats: dict, num_predictions: int = 60) -> list:
    """Predict future candles using the trained model"""
    predictions = []
    current_sequence = input_sequence.copy()

    last_candle = input_sequence[-1]
    last_close = last_candle[3]  # close is index 3

    for i in range(num_predictions):
        # Reshape for model: (1, sequence_length, features)
        model_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])

        # Predict next close price (normalized)
        predicted_close_norm = model.predict(model_input, verbose=0)[0][0]

        # Ensure prediction is within valid range
        predicted_close_norm = max(0, min(1, predicted_close_norm))

        # Create a predicted candle
        # Use the predicted close as the basis, add some realistic OHLC spread
        volatility = 0.002  # 0.2% typical volatility for 1-min candle

        pred_open = last_close
        pred_close = predicted_close_norm
        pred_high = max(pred_open, pred_close) + abs(pred_close - pred_open) * 0.3
        pred_low = min(pred_open, pred_close) - abs(pred_close - pred_open) * 0.3

        # Ensure high >= max(open, close) and low <= min(open, close)
        pred_high = max(pred_high, pred_open, pred_close)
        pred_low = min(pred_low, pred_open, pred_close)

        # Keep volume similar to recent average
        avg_volume = np.mean(current_sequence[-10:, 4])
        pred_volume = avg_volume * (0.8 + np.random.random() * 0.4)

        pred_candle = np.array([pred_open, pred_high, pred_low, pred_close, pred_volume])

        # Denormalize for output
        predictions.append({
            'open': float(denormalize_price(pred_open, stats)),
            'high': float(denormalize_price(pred_high, stats)),
            'low': float(denormalize_price(pred_low, stats)),
            'close': float(denormalize_price(pred_close, stats)),
            'volume': float(pred_volume * (stats['max_volume'] - stats['min_volume']) + stats['min_volume'])
        })

        # Update sequence: remove oldest, add new prediction
        current_sequence = np.vstack([current_sequence[1:], pred_candle])
        last_close = pred_close

    return predictions

def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: predict.py <category> <asset> [candles_json]'}))
        sys.exit(1)

    category = sys.argv[1]
    asset = sys.argv[2]

    # Read candles from stdin or file
    if len(sys.argv) > 3:
        candles_json = sys.argv[3]
    else:
        candles_json = sys.stdin.read()

    try:
        input_data = json.loads(candles_json)
        candles = input_data.get('candles', input_data)
        output_minutes = input_data.get('outputMinutes', 60)

        # Load model
        model, stats = load_model_and_stats(category, asset)

        sequence_length = stats.get('sequence_length', 60)

        if len(candles) < sequence_length:
            print(json.dumps({
                'error': f'Need at least {sequence_length} candles, got {len(candles)}'
            }))
            sys.exit(1)

        # Take the last sequence_length candles
        recent_candles = candles[-sequence_length:]

        # Normalize
        normalized = normalize_candles(recent_candles, stats)

        # Predict
        predictions = predict_future(model, normalized, stats, output_minutes)

        # Add timestamps
        last_timestamp = candles[-1].get('timestamp', 0)
        for i, pred in enumerate(predictions):
            pred['timestamp'] = last_timestamp + (i + 1) * 60000  # +1 minute each

        # Calculate summary
        current_price = candles[-1]['close']
        final_price = predictions[-1]['close']
        percent_change = ((final_price - current_price) / current_price) * 100

        result = {
            'predictions': predictions,
            'summary': {
                'currentPrice': current_price,
                'predictedPrice': final_price,
                'percentChange': percent_change,
                'direction': 'up' if percent_change > 0.1 else 'down' if percent_change < -0.1 else 'neutral'
            },
            'model': {
                'asset': asset,
                'category': category,
                'sequenceLength': sequence_length,
                'trainedCandles': stats.get('total_candles', 0)
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
