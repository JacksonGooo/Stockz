#!/usr/bin/env python3
"""
Backtesting Script for BTC Autoregressive Model
Tests the model on random historical timestamps and measures accuracy
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES
from data_loader import load_all_btc_candles

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'autoregressive_btc')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')

# Settings
SEQUENCE_LENGTH = 30
PREDICTION_HORIZON = 60
NUM_TESTS = 50  # Number of random backtests to run


def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_features(candles_df):
    """Calculate features and normalize"""
    features = calculate_all_features(candles_df)
    normalized = rolling_zscore(features, window=60)
    normalized = normalized.fillna(0)
    return normalized[FEATURE_NAMES].values


def predict_sequence(model, features, num_predictions=60):
    """Generate autoregressive predictions"""
    current_sequence = features[-SEQUENCE_LENGTH:].copy()
    predictions = []

    for i in range(num_predictions):
        model_input = current_sequence.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        pred_returns = model.predict(model_input, verbose=0)[0]

        predictions.append({
            'open_return': float(pred_returns[0]),
            'high_return': float(pred_returns[1]),
            'low_return': float(pred_returns[2]),
            'close_return': float(pred_returns[3]),
            'volume_change': float(pred_returns[4])
        })

        # Create new feature row
        new_features = current_sequence[-1].copy()
        new_features[0] = pred_returns[0]
        new_features[1] = pred_returns[1]
        new_features[2] = pred_returns[2]
        new_features[3] = pred_returns[3]
        new_features[4] = pred_returns[4]

        current_sequence = np.vstack([current_sequence[1:], new_features])

    return predictions


def returns_to_prices(predictions, start_price):
    """Convert predicted returns to cumulative price"""
    prices = [start_price]
    current_price = start_price

    for pred in predictions:
        current_price = current_price * (1 + pred['close_return'])
        prices.append(current_price)

    return prices[1:]  # Exclude start price


def run_backtest(model, all_candles, test_idx):
    """Run a single backtest at a specific index"""
    # Need SEQUENCE_LENGTH + 60 for features calculation buffer
    # Plus PREDICTION_HORIZON for actual data to compare
    buffer = 100

    if test_idx < buffer or test_idx >= len(all_candles) - PREDICTION_HORIZON:
        return None

    # Get candles for feature calculation (need more than just 30 for rolling calculations)
    start_idx = test_idx - buffer
    end_idx = test_idx

    input_candles = all_candles.iloc[start_idx:end_idx].copy()
    actual_future = all_candles.iloc[end_idx:end_idx + PREDICTION_HORIZON].copy()

    if len(actual_future) < PREDICTION_HORIZON:
        return None

    # Prepare features
    features = prepare_features(input_candles)

    if len(features) < SEQUENCE_LENGTH:
        return None

    # Get starting price
    start_price = input_candles.iloc[-1]['close']
    start_time = input_candles.iloc[-1]['timestamp']

    # Predict
    predictions = predict_sequence(model, features, PREDICTION_HORIZON)
    predicted_prices = returns_to_prices(predictions, start_price)

    # Get actual prices
    actual_prices = actual_future['close'].values

    # Calculate metrics
    predicted_final = predicted_prices[-1]
    actual_final = actual_prices[-1]

    predicted_change = (predicted_final - start_price) / start_price * 100
    actual_change = (actual_final - start_price) / start_price * 100

    predicted_direction = 'UP' if predicted_change > 0 else 'DOWN'
    actual_direction = 'UP' if actual_change > 0 else 'DOWN'
    direction_correct = predicted_direction == actual_direction

    # Price accuracy (how close was the prediction)
    price_error = abs(predicted_final - actual_final)
    price_error_percent = price_error / start_price * 100

    # Return error (how close was the predicted change)
    return_error = abs(predicted_change - actual_change)

    return {
        'timestamp': datetime.fromtimestamp(start_time / 1000).strftime('%Y-%m-%d %H:%M'),
        'start_price': start_price,
        'predicted_final': predicted_final,
        'actual_final': actual_final,
        'predicted_change': predicted_change,
        'actual_change': actual_change,
        'predicted_direction': predicted_direction,
        'actual_direction': actual_direction,
        'direction_correct': direction_correct,
        'price_error_percent': price_error_percent,
        'return_error': return_error,
    }


def main():
    print("=" * 60)
    print("BTC AUTOREGRESSIVE MODEL BACKTESTING")
    print("=" * 60)
    print(f"Testing on {NUM_TESTS} random historical timestamps")
    print(f"Each test: 30 min input → 60 min prediction")
    print()

    # Load model
    print("Loading model...")
    model = load_model()

    # Load all BTC data
    print("Loading BTC historical data...")
    all_candles = load_all_btc_candles()
    print(f"Loaded {len(all_candles):,} candles")
    print()

    # Run backtests at random indices
    buffer = 150
    valid_range = range(buffer, len(all_candles) - PREDICTION_HORIZON - 10)
    test_indices = random.sample(list(valid_range), min(NUM_TESTS, len(valid_range)))

    results = []

    print("Running backtests...")
    print("-" * 60)

    for i, idx in enumerate(test_indices):
        result = run_backtest(model, all_candles, idx)

        if result:
            results.append(result)

            # Print individual result
            direction_mark = "✓" if result['direction_correct'] else "✗"
            print(f"Test {i+1}: {result['timestamp']}")
            print(f"  Predicted: {result['predicted_change']:+.2f}% ({result['predicted_direction']})")
            print(f"  Actual:    {result['actual_change']:+.2f}% ({result['actual_direction']}) [{direction_mark}]")
            print(f"  Return Error: {result['return_error']:.2f}%")
            print()

    # Calculate summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if results:
        direction_accuracy = sum(r['direction_correct'] for r in results) / len(results) * 100
        avg_return_error = np.mean([r['return_error'] for r in results])
        avg_price_error = np.mean([r['price_error_percent'] for r in results])

        # Calculate by predicted direction
        up_predictions = [r for r in results if r['predicted_direction'] == 'UP']
        down_predictions = [r for r in results if r['predicted_direction'] == 'DOWN']

        up_accuracy = sum(r['direction_correct'] for r in up_predictions) / len(up_predictions) * 100 if up_predictions else 0
        down_accuracy = sum(r['direction_correct'] for r in down_predictions) / len(down_predictions) * 100 if down_predictions else 0

        print(f"Total Tests: {len(results)}")
        print()
        print(f"DIRECTION ACCURACY: {direction_accuracy:.1f}%")
        print(f"  - When predicting UP:   {up_accuracy:.1f}% ({len(up_predictions)} tests)")
        print(f"  - When predicting DOWN: {down_accuracy:.1f}% ({len(down_predictions)} tests)")
        print()
        print(f"AVERAGE ERRORS:")
        print(f"  - Return Error: {avg_return_error:.2f}% (diff between predicted and actual % change)")
        print(f"  - Price Error:  {avg_price_error:.2f}% (diff in final price)")
        print()

        # Distribution of actual changes when prediction was correct
        correct_results = [r for r in results if r['direction_correct']]
        if correct_results:
            correct_changes = [abs(r['actual_change']) for r in correct_results]
            print(f"When direction was CORRECT:")
            print(f"  - Avg actual move: {np.mean(correct_changes):.2f}%")
            print(f"  - Max actual move: {max(correct_changes):.2f}%")

        # Save results
        output_path = os.path.join(MODEL_DIR, 'backtest_results.json')
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': len(results),
                    'direction_accuracy': direction_accuracy,
                    'avg_return_error': avg_return_error,
                    'avg_price_error': avg_price_error,
                    'up_accuracy': up_accuracy,
                    'down_accuracy': down_accuracy,
                },
                'individual_results': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No valid test results!")


if __name__ == '__main__':
    main()
