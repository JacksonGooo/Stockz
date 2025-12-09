"""
BTC LSTM Prediction Tester
Tests model accuracy on historical data

Usage:
    python test_predictions.py              # Test with default settings (1 week ago)
    python test_predictions.py --days 7     # Test from N days ago
    python test_predictions.py --samples 10 # Run N random test samples
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Paths
DATA_DIR = Path(__file__).parent.parent / 'Data'
MODELS_DIR = Path(__file__).parent.parent / 'models'


def load_model_and_stats(asset='btc'):
    """Load trained model and normalization stats"""
    model_path = MODELS_DIR / f'{asset}_lstm.keras'
    stats_path = MODELS_DIR / f'{asset}_lstm_stats.json'

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

    # Load from week folders
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

    # Remove duplicates and sort
    unique = {c['timestamp']: c for c in all_candles}
    sorted_candles = sorted(unique.values(), key=lambda x: x['timestamp'])

    return sorted_candles


def normalize_candle(candle, stats):
    """Normalize a single candle"""
    price_range = stats['max_price'] - stats['min_price'] or 1
    volume_range = stats['max_volume'] - stats['min_volume'] or 1

    return [
        (candle['open'] - stats['min_price']) / price_range,
        (candle['high'] - stats['min_price']) / price_range,
        (candle['low'] - stats['min_price']) / price_range,
        (candle['close'] - stats['min_price']) / price_range,
        (candle['volume'] - stats['min_volume']) / volume_range,
    ]


def denormalize_price(normalized_price, stats):
    """Convert normalized price back to actual price"""
    price_range = stats['max_price'] - stats['min_price'] or 1
    return normalized_price * price_range + stats['min_price']


def make_prediction(model, stats, candles, sequence_length):
    """Make a prediction from a sequence of candles"""
    if len(candles) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} candles")

    # Take the last sequence_length candles
    sequence = candles[-sequence_length:]

    # Normalize
    normalized = [normalize_candle(c, stats) for c in sequence]

    # Reshape for model: (1, sequence_length, 5)
    input_data = np.array([normalized], dtype=np.float32)

    # Predict
    prediction = model.predict(input_data, verbose=0)[0][0]

    # Denormalize
    predicted_price = denormalize_price(prediction, stats)

    return predicted_price


def find_candles_at_time(all_candles, target_time):
    """Find candles around a specific timestamp"""
    target_ts = target_time.timestamp() * 1000  # Convert to milliseconds

    # Find nearest candle
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
    # Get sequence of candles leading up to test_idx
    if test_idx < sequence_length:
        return None

    sequence = all_candles[test_idx - sequence_length:test_idx]
    actual_candle = all_candles[test_idx]

    # Make prediction
    predicted_price = make_prediction(model, stats, sequence, sequence_length)
    actual_price = actual_candle['close']

    # Calculate error
    error = predicted_price - actual_price
    error_percent = (error / actual_price) * 100

    # Determine direction
    prev_close = sequence[-1]['close']
    actual_direction = 'UP' if actual_price > prev_close else 'DOWN'
    predicted_direction = 'UP' if predicted_price > prev_close else 'DOWN'
    direction_correct = actual_direction == predicted_direction

    return {
        'timestamp': actual_candle['timestamp'],
        'datetime': datetime.fromtimestamp(actual_candle['timestamp'] / 1000).isoformat(),
        'prev_close': prev_close,
        'predicted': predicted_price,
        'actual': actual_price,
        'error': error,
        'error_percent': error_percent,
        'actual_direction': actual_direction,
        'predicted_direction': predicted_direction,
        'direction_correct': direction_correct,
    }


def run_tests(days_ago=7, num_samples=20):
    """Run multiple prediction tests"""
    print('=' * 70)
    print('BTC LSTM Prediction Accuracy Test')
    print('=' * 70)
    print()

    # Load model and data
    print('Loading model...')
    model, stats = load_model_and_stats('btc')
    sequence_length = stats.get('sequence_length', 30)
    print(f"Model trained on: {stats.get('trained_at', 'unknown')}")
    print(f"Sequence length: {sequence_length} minutes")
    print(f"Price range: ${stats['min_price']:,.2f} - ${stats['max_price']:,.2f}")
    print()

    print('Loading BTC data...')
    all_candles = load_btc_data()
    print(f"Loaded {len(all_candles):,} candles")
    print()

    # Find test period (days_ago)
    target_time = datetime.now() - timedelta(days=days_ago)
    test_start_idx = find_candles_at_time(all_candles, target_time)

    if test_start_idx < sequence_length + num_samples:
        print(f"Error: Not enough data for testing from {days_ago} days ago")
        return

    # Get test date range info
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

    # Print individual results
    print('-' * 70)
    print(f'{"Time":<20} {"Prev":>10} {"Pred":>10} {"Actual":>10} {"Error%":>8} {"Dir":>5}')
    print('-' * 70)

    for r in results:
        dt = datetime.fromisoformat(r['datetime'])
        time_str = dt.strftime('%m/%d %H:%M')
        dir_symbol = 'Y' if r['direction_correct'] else 'N'
        print(f"{time_str:<20} ${r['prev_close']:>9,.0f} ${r['predicted']:>9,.0f} ${r['actual']:>9,.0f} {r['error_percent']:>7.2f}% {dir_symbol:>5}")

    print('-' * 70)

    # Summary statistics
    errors = [abs(r['error_percent']) for r in results]
    directions_correct = sum(1 for r in results if r['direction_correct'])

    avg_error = np.mean(errors)
    max_error = max(errors)
    min_error = min(errors)
    direction_accuracy = (directions_correct / len(results)) * 100

    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f"Samples tested:      {len(results)}")
    print(f"Average error:       {avg_error:.2f}%")
    print(f"Min error:           {min_error:.2f}%")
    print(f"Max error:           {max_error:.2f}%")
    print(f"Direction accuracy:  {directions_correct}/{len(results)} ({direction_accuracy:.1f}%)")
    print()

    # Interpret results
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
        'results': results,
    }


def test_specific_period(start_time_str, num_minutes=30):
    """Test predictions for a specific time period"""
    print('=' * 70)
    print('BTC Prediction Test - Specific Period')
    print('=' * 70)
    print()

    # Parse start time
    try:
        start_time = datetime.fromisoformat(start_time_str)
    except ValueError:
        print(f"Invalid time format: {start_time_str}")
        print("Use format: YYYY-MM-DD HH:MM")
        return

    # Load model and data
    model, stats = load_model_and_stats('btc')
    sequence_length = stats.get('sequence_length', 30)
    all_candles = load_btc_data()

    # Find starting index
    start_idx = find_candles_at_time(all_candles, start_time)

    print(f"Testing period: {start_time.strftime('%Y-%m-%d %H:%M')} for {num_minutes} minutes")
    print(f"Model sequence length: {sequence_length}")
    print()

    # Run predictions
    results = []
    for i in range(num_minutes):
        idx = start_idx + sequence_length + i
        if idx >= len(all_candles):
            break
        result = run_test(model, stats, all_candles, idx, sequence_length)
        if result:
            results.append(result)

    if results:
        # Show results
        for r in results[:10]:  # Show first 10
            dt = datetime.fromisoformat(r['datetime'])
            print(f"{dt.strftime('%H:%M')} - Predicted: ${r['predicted']:,.0f} | Actual: ${r['actual']:,.0f} | Error: {r['error_percent']:.2f}%")

        if len(results) > 10:
            print(f"... and {len(results) - 10} more")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test BTC LSTM predictions')
    parser.add_argument('--days', type=int, default=7, help='Days ago to test from')
    parser.add_argument('--samples', type=int, default=30, help='Number of test samples')
    parser.add_argument('--time', type=str, help='Specific start time (YYYY-MM-DD HH:MM)')

    args = parser.parse_args()

    if args.time:
        test_specific_period(args.time, args.samples)
    else:
        run_tests(days_ago=args.days, num_samples=args.samples)
