"""
Multi-Test Script for BTC Returns Model
Runs multiple tests across different time periods to evaluate consistency

Usage:
    python test_returns_multi.py              # Run 5 test batches
    python test_returns_multi.py --runs 10    # Run 10 test batches
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

    sequence = candles[-(sequence_length + 1):]
    returns_features = calculate_returns_for_candles(sequence, max_volume)

    if len(returns_features) < sequence_length:
        raise ValueError("Not enough data to calculate returns")

    input_data = np.array([returns_features[-sequence_length:]], dtype=np.float32)
    predicted_normalized = model.predict(input_data, verbose=0)[0][0]
    predicted_return = predicted_normalized * stats['return_scale']
    last_close = candles[-1]['close']
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


def run_single_test_batch(model, stats, all_candles, days_ago, num_samples=30):
    """Run a single batch of predictions"""
    sequence_length = stats['sequence_length']

    target_time = datetime.now() - timedelta(days=days_ago)
    test_start_idx = find_candles_at_time(all_candles, target_time)

    if test_start_idx < sequence_length + num_samples + 5:
        return None

    results = []
    for i in range(num_samples):
        idx = test_start_idx + i
        if idx >= len(all_candles) or idx < sequence_length + 2:
            continue

        sequence = all_candles[idx - sequence_length - 1:idx]
        actual_candle = all_candles[idx]

        try:
            predicted_price, predicted_return = make_prediction(model, stats, sequence)
            actual_price = actual_candle['close']
            prev_close = sequence[-1]['close']
            actual_return = (actual_price - prev_close) / prev_close * 100

            error_percent = abs((predicted_price - actual_price) / actual_price) * 100
            actual_direction = 'UP' if actual_return > 0 else 'DOWN'
            predicted_direction = 'UP' if predicted_return > 0 else 'DOWN'
            direction_correct = actual_direction == predicted_direction

            results.append({
                'error_percent': error_percent,
                'direction_correct': direction_correct,
                'predicted_return': predicted_return,
                'actual_return': actual_return,
            })
        except Exception:
            continue

    if not results:
        return None

    errors = [r['error_percent'] for r in results]
    directions_correct = sum(1 for r in results if r['direction_correct'])

    return {
        'days_ago': days_ago,
        'samples': len(results),
        'avg_error': np.mean(errors),
        'min_error': min(errors),
        'max_error': max(errors),
        'direction_accuracy': (directions_correct / len(results)) * 100,
        'perfect_count': sum(1 for e in errors if e < 0.01),
        'excellent_count': sum(1 for e in errors if e < 0.1),
        'good_count': sum(1 for e in errors if e < 0.5),
    }


def run_multi_tests(num_runs=5, samples_per_run=30):
    """Run multiple test batches across different time periods"""
    print('=' * 70)
    print('BTC Returns Model - MULTI-TEST CONSISTENCY CHECK')
    print('=' * 70)
    print()

    # Load model
    print('Loading returns model...')
    model, stats = load_model_and_stats('btc')
    print(f"Model trained on: {stats.get('trained_at', 'unknown')}")
    print(f"Sequence length: {stats['sequence_length']} minutes")
    print()

    # Load data
    print('Loading BTC data...')
    all_candles = load_btc_data()
    print(f"Loaded {len(all_candles):,} candles")
    print()

    # Run tests at different time periods
    test_days = [3, 5, 7, 10, 14, 21, 30][:num_runs]

    print(f"Running {len(test_days)} test batches ({samples_per_run} samples each)...")
    print()
    print('-' * 80)
    print(f'{"Days Ago":<10} {"Samples":<10} {"Avg Err%":<12} {"Min Err%":<12} {"Max Err%":<12} {"Dir Acc":<10} {"<0.01%":<8}')
    print('-' * 80)

    all_results = []
    total_perfect = 0
    total_excellent = 0
    total_good = 0
    total_samples = 0
    all_errors = []
    all_directions = []

    for days in test_days:
        result = run_single_test_batch(model, stats, all_candles, days, samples_per_run)
        if result:
            all_results.append(result)
            total_perfect += result['perfect_count']
            total_excellent += result['excellent_count']
            total_good += result['good_count']
            total_samples += result['samples']
            all_errors.append(result['avg_error'])
            all_directions.append(result['direction_accuracy'])

            print(f"{result['days_ago']:<10} {result['samples']:<10} {result['avg_error']:<12.4f} {result['min_error']:<12.4f} {result['max_error']:<12.4f} {result['direction_accuracy']:<10.1f}% {result['perfect_count']:<8}")

    print('-' * 80)
    print()

    if not all_results:
        print("No test results generated")
        return

    # Overall summary
    print('=' * 70)
    print('OVERALL SUMMARY')
    print('=' * 70)
    print(f"Total test batches:     {len(all_results)}")
    print(f"Total samples tested:   {total_samples}")
    print()
    print(f"Average error:          {np.mean(all_errors):.4f}%")
    print(f"Std deviation:          {np.std(all_errors):.4f}%")
    print(f"Best batch:             {min(all_errors):.4f}%")
    print(f"Worst batch:            {max(all_errors):.4f}%")
    print()
    print(f"Avg direction accuracy: {np.mean(all_directions):.1f}%")
    print()
    print(f"Predictions <0.01% err: {total_perfect}/{total_samples} ({100*total_perfect/total_samples:.1f}%)")
    print(f"Predictions <0.1% err:  {total_excellent}/{total_samples} ({100*total_excellent/total_samples:.1f}%)")
    print(f"Predictions <0.5% err:  {total_good}/{total_samples} ({100*total_good/total_samples:.1f}%)")
    print()

    # Final verdict
    avg_error = np.mean(all_errors)
    avg_direction = np.mean(all_directions)
    consistency = np.std(all_errors) < 0.1  # Low variance across batches

    print('=' * 70)
    print('VERDICT')
    print('=' * 70)

    if avg_error < 0.01:
        print("[+] PERFECT: Average error <0.01%")
    elif avg_error < 0.1:
        print("[+] EXCELLENT: Average error <0.1%")
    elif avg_error < 0.5:
        print("[+] GOOD: Average error <0.5%")
    elif avg_error < 1.0:
        print("[=] MODERATE: Average error <1%")
    else:
        print("[-] NEEDS IMPROVEMENT: Average error >1%")

    if avg_direction >= 55:
        print("[+] Direction prediction: GOOD (>55%)")
    elif avg_direction >= 50:
        print("[=] Direction prediction: NEUTRAL (~50%)")
    else:
        print("[-] Direction prediction: POOR (<50%)")

    if consistency:
        print("[+] Consistency: STABLE (low variance across time periods)")
    else:
        print("[=] Consistency: VARIABLE (performance varies by time period)")

    print()

    return {
        'num_batches': len(all_results),
        'total_samples': total_samples,
        'avg_error': np.mean(all_errors),
        'avg_direction': avg_direction,
        'perfect_rate': 100*total_perfect/total_samples,
        'excellent_rate': 100*total_excellent/total_samples,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Multi-test BTC returns model')
    parser.add_argument('--runs', type=int, default=5, help='Number of test batches')
    parser.add_argument('--samples', type=int, default=30, help='Samples per batch')

    args = parser.parse_args()
    run_multi_tests(num_runs=args.runs, samples_per_run=args.samples)
