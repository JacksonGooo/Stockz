"""
Test script for the returns-based LSTM model
Tests model accuracy on historical data by predicting % returns

Usage:
    python test_returns_model.py              # Test from 7 days ago
    python test_returns_model.py --days 7     # Test from N days ago
    python test_returns_model.py --samples 30 # Run N test samples
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Paths
DATA_DIR = Path(__file__).parent.parent / 'Data'
MODELS_DIR = Path(__file__).parent.parent / 'models'


def load_model_and_stats(asset='btc'):
    """Load trained returns model and stats"""
    model_path = MODELS_DIR / f'{asset}_lstm_returns.keras'
    stats_path = MODELS_DIR / f'{asset}_lstm_returns_stats.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    model = tf.keras.models.load_model(str(model_path))

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return model, stats


def load_btc_data():
    """Load all BTC candle data"""
    asset_dir = DATA_DIR / 'Crypto' / 'BTC'

    if not asset_dir.exists():
        raise FileNotFoundError(f"No data found for Crypto/BTC")

    all_candles = []
    week_dirs = sorted([d for d in asset_dir.iterdir() if d.name.startswith('week_')])

    for week_dir in week_dirs:
        json_files = sorted(week_dir.glob('*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_candles.extend(data)
            except (json.JSONDecodeError, IOError):
                continue

    unique = {c['timestamp']: c for c in all_candles}
    sorted_candles = sorted(unique.values(), key=lambda x: x['timestamp'])

    return sorted_candles


def calculate_returns_for_candles(candles, max_volume):
    """Calculate returns features for a sequence of candles"""
    returns_data = []

    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]

        if prev['close'] > 0:
            close_return = (curr['close'] - prev['close']) / prev['close'] * 100
            high_return = (curr['high'] - prev['close']) / prev['close'] * 100
            low_return = (curr['low'] - prev['close']) / prev['close'] * 100
            open_return = (curr['open'] - prev['close']) / prev['close'] * 100
            log_volume = np.log1p(curr['volume']) if curr['volume'] > 0 else 0
            price_range = (curr['high'] - curr['low']) / prev['close'] * 100

            # Normalize
            returns_data.append([
                np.clip(open_return / 5, -1, 1),
                np.clip(high_return / 5, -1, 1),
                np.clip(low_return / 5, -1, 1),
                np.clip(close_return / 5, -1, 1),
                log_volume / max_volume if max_volume > 0 else 0,
                np.clip(price_range / 5, 0, 1),
            ])

    return returns_data


def make_prediction(model, stats, candles):
    """Make a prediction from a sequence of candles"""
    sequence_length = stats['sequence_length']
    max_volume = stats['max_volume']

    if len(candles) < sequence_length + 1:
        raise ValueError(f"Need at least {sequence_length + 1} candles")

    # Take sequence of candles (need +1 because we need pairs to compute returns)
    sequence = candles[-(sequence_length + 1):]

    # Calculate returns
    returns_features = calculate_returns_for_candles(sequence, max_volume)

    if len(returns_features) < sequence_length:
        raise ValueError("Not enough data to calculate returns")

    # Reshape for model: (1, sequence_length, features)
    input_data = np.array([returns_features[-sequence_length:]], dtype=np.float32)

    # Predict
    predicted_normalized = model.predict(input_data, verbose=0)[0][0]

    # De-normalize (multiply by scale factor)
    predicted_return = predicted_normalized * stats['return_scale']

    # Get the last close price
    last_close = candles[-1]['close']

    # Convert return to predicted price
    predicted_price = last_close * (1 + predicted_return / 100)

    return predicted_price, predicted_return


def find_candles_at_time(all_candles, target_time):
    """Find candles around a specific timestamp"""
    target_ts = target_time.timestamp() * 1000

    nearest_idx = 0
    min_diff = float('inf')

    for i, c in enumerate(all_candles):
        diff = abs(c['timestamp'] - target_ts)
        if diff < min_diff:
            min_diff = diff
            nearest_idx = i

    return nearest_idx


def run_test(model, stats, all_candles, test_idx, sequence_length):
    """Run a single prediction test"""
    # Need sequence_length + 1 candles before test_idx to compute returns
    if test_idx < sequence_length + 2:
        return None

    # Get sequence + 1 extra for return calculation
    sequence = all_candles[test_idx - sequence_length - 1:test_idx]
    actual_candle = all_candles[test_idx]

    # Make prediction
    predicted_price, predicted_return = make_prediction(model, stats, sequence)
    actual_price = actual_candle['close']
    prev_close = sequence[-1]['close']

    # Calculate actual return
    actual_return = (actual_price - prev_close) / prev_close * 100

    # Calculate price error
    error = predicted_price - actual_price
    error_percent = (error / actual_price) * 100

    # Direction prediction
    actual_direction = 'UP' if actual_return > 0 else 'DOWN'
    predicted_direction = 'UP' if predicted_return > 0 else 'DOWN'
    direction_correct = actual_direction == predicted_direction

    return {
        'timestamp': actual_candle['timestamp'],
        'datetime': datetime.fromtimestamp(actual_candle['timestamp'] / 1000).isoformat(),
        'prev_close': prev_close,
        'predicted': predicted_price,
        'actual': actual_price,
        'predicted_return': predicted_return,
        'actual_return': actual_return,
        'error': error,
        'error_percent': error_percent,
        'actual_direction': actual_direction,
        'predicted_direction': predicted_direction,
        'direction_correct': direction_correct,
    }


def run_tests(days_ago=7, num_samples=30):
    """Run multiple prediction tests"""
    print('=' * 70)
    print('BTC Returns Model - Prediction Accuracy Test')
    print('=' * 70)
    print()

    # Load model
    print('Loading returns model...')
    model, stats = load_model_and_stats('btc')
    sequence_length = stats['sequence_length']
    print(f"Model trained on: {stats.get('trained_at', 'unknown')}")
    print(f"Sequence length: {sequence_length} minutes")
    print(f"Return scale: {stats['return_scale']}")
    print()

    # Load data
    print('Loading BTC data...')
    all_candles = load_btc_data()
    print(f"Loaded {len(all_candles):,} candles")
    print()

    # Find test period
    target_time = datetime.now() - timedelta(days=days_ago)
    test_start_idx = find_candles_at_time(all_candles, target_time)

    if test_start_idx < sequence_length + num_samples + 5:
        print(f"Error: Not enough data for testing from {days_ago} days ago")
        return

    start_candle = all_candles[test_start_idx]
    start_dt = datetime.fromtimestamp(start_candle['timestamp'] / 1000)
    print(f"Testing from: {start_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"Running {num_samples} predictions...")
    print()

    # Run tests
    results = []
    for i in range(num_samples):
        idx = test_start_idx + i
        if idx >= len(all_candles):
            break
        result = run_test(model, stats, all_candles, idx, sequence_length)
        if result:
            results.append(result)

    if not results:
        print("No test results generated")
        return

    # Print results
    print('-' * 80)
    print(f'{"Time":<15} {"Prev":>10} {"Pred":>10} {"Actual":>10} {"PredRet":>8} {"ActRet":>8} {"Dir":>5}')
    print('-' * 80)

    for r in results:
        dt = datetime.fromisoformat(r['datetime'])
        time_str = dt.strftime('%m/%d %H:%M')
        dir_symbol = 'Y' if r['direction_correct'] else 'N'
        print(f"{time_str:<15} ${r['prev_close']:>9,.0f} ${r['predicted']:>9,.0f} ${r['actual']:>9,.0f} {r['predicted_return']:>7.3f}% {r['actual_return']:>7.3f}% {dir_symbol:>5}")

    print('-' * 80)

    # Summary
    errors = [abs(r['error_percent']) for r in results]
    return_errors = [abs(r['predicted_return'] - r['actual_return']) for r in results]
    directions_correct = sum(1 for r in results if r['direction_correct'])

    avg_error = np.mean(errors)
    avg_return_error = np.mean(return_errors)
    direction_accuracy = (directions_correct / len(results)) * 100

    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f"Samples tested:      {len(results)}")
    print(f"Avg price error:     {avg_error:.3f}%")
    print(f"Avg return error:    {avg_return_error:.4f}%")
    print(f"Direction accuracy:  {directions_correct}/{len(results)} ({direction_accuracy:.1f}%)")
    print()

    # Interpret
    if direction_accuracy >= 55:
        print("[+] Direction prediction: GOOD (>55% accurate)")
    elif direction_accuracy >= 50:
        print("[=] Direction prediction: NEUTRAL (~50%)")
    else:
        print("[-] Direction prediction: POOR (<50%)")

    if avg_error < 0.01:
        print("[+] Price error: PERFECT (<0.01%)")
    elif avg_error < 0.1:
        print("[+] Price error: EXCELLENT (<0.1%)")
    elif avg_error < 0.5:
        print("[+] Price error: GOOD (<0.5%)")
    elif avg_error < 1.0:
        print("[=] Price error: MODERATE (<1%)")
    else:
        print("[!] Price error: HIGH (>1%)")

    print()

    return {
        'num_samples': len(results),
        'avg_error': avg_error,
        'direction_accuracy': direction_accuracy,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test BTC returns model')
    parser.add_argument('--days', type=int, default=7, help='Days ago to test from')
    parser.add_argument('--samples', type=int, default=30, help='Number of test samples')

    args = parser.parse_args()
    run_tests(days_ago=args.days, num_samples=args.samples)
