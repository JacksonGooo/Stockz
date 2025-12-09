"""
60-Minute BTC Price Prediction Script
Loads trained model and generates 60-minute price forecasts

Usage:
  python predict_60min.py          # Predict from latest data
  python predict_60min.py --test   # Test on random historical timestamps
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import sys
from datetime import datetime

# Force CPU
tf.config.set_visible_devices([], 'GPU')

from data_loader import load_all_btc_candles, SEQUENCE_LENGTH, PREDICTION_HORIZON
from features import calculate_all_features, rolling_zscore, NUM_FEATURES, FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'autoregressive_btc')


def load_trained_model():
    """Load the trained autoregressive model"""
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first with: python train_autoregressive_btc.py")
        return None

    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def prepare_latest_sequence(df):
    """Prepare the most recent sequence for prediction"""
    # Calculate features
    features_df = calculate_all_features(df)
    normalized_df = rolling_zscore(features_df, window=60).fillna(0)
    feature_data = normalized_df.values.astype(np.float32)

    # Get the last SEQUENCE_LENGTH candles
    if len(feature_data) < SEQUENCE_LENGTH:
        print(f"Not enough data: {len(feature_data)} candles, need {SEQUENCE_LENGTH}")
        return None, None

    latest_sequence = feature_data[-SEQUENCE_LENGTH:]
    latest_timestamp = df['timestamp'].iloc[-1]
    latest_price = df['close'].iloc[-1]

    return latest_sequence[np.newaxis, :, :], {
        'timestamp': latest_timestamp,
        'datetime': datetime.fromtimestamp(latest_timestamp / 1000),
        'current_price': latest_price
    }


def predict_60_minutes(model, input_sequence):
    """
    Generate 60-minute prediction using autoregressive approach
    """
    predictions = []
    current_seq = input_sequence.copy()

    # Feature indices for OHLCV returns
    ohlcv_features = ['open_return', 'high_return', 'low_return', 'close_return', 'volume_change']
    ohlcv_indices = [FEATURE_NAMES.index(f) for f in ohlcv_features]

    for i in range(PREDICTION_HORIZON):
        # Predict next candle
        pred = model.predict(current_seq, verbose=0)[0]
        predictions.append(pred)

        # Create new feature vector
        new_features = current_seq[0, -1, :].copy()

        # Update OHLCV returns
        for j, idx in enumerate(ohlcv_indices):
            new_features[idx] = pred[j]

        # Shift and append
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_features

    return np.array(predictions)


def analyze_prediction(predictions, current_price):
    """
    Analyze the 60-minute prediction trajectory
    """
    # Extract close returns
    close_returns = predictions[:, 3]  # Index 3 is close_return

    # Calculate cumulative return
    cumulative_return = np.sum(close_returns)

    # Calculate price trajectory
    price_trajectory = [current_price]
    for ret in close_returns:
        next_price = price_trajectory[-1] * (1 + ret)
        price_trajectory.append(next_price)

    # Predicted final price
    predicted_price = price_trajectory[-1]
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100

    # Determine signal
    if cumulative_return > 0.001:  # > 0.1%
        signal = "BUY"
        confidence = min(abs(cumulative_return) * 1000, 100)  # Scale to 0-100
    elif cumulative_return < -0.001:  # < -0.1%
        signal = "SELL"
        confidence = min(abs(cumulative_return) * 1000, 100)
    else:
        signal = "HOLD"
        confidence = 100 - abs(cumulative_return) * 1000

    # Find min/max in trajectory
    min_price = min(price_trajectory)
    max_price = max(price_trajectory)
    min_time = price_trajectory.index(min_price)
    max_time = price_trajectory.index(max_price)

    return {
        'current_price': current_price,
        'predicted_price': predicted_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'cumulative_return': cumulative_return,
        'signal': signal,
        'confidence': confidence,
        'trajectory': price_trajectory,
        'min_price': min_price,
        'max_price': max_price,
        'min_time': min_time,
        'max_time': max_time,
        'volatility': np.std(close_returns) * 100
    }


def print_prediction(analysis, context):
    """Print prediction results in a nice format"""
    print("\n" + "=" * 60)
    print("60-MINUTE BTC PRICE PREDICTION")
    print("=" * 60)

    print(f"\nCurrent Time: {context['datetime']}")
    print(f"Current Price: ${analysis['current_price']:,.2f}")

    print("\n--- 60-MINUTE FORECAST ---")
    print(f"Predicted Price: ${analysis['predicted_price']:,.2f}")
    print(f"Price Change: ${analysis['price_change']:+,.2f} ({analysis['price_change_pct']:+.3f}%)")

    print(f"\nSignal: {analysis['signal']}")
    print(f"Confidence: {analysis['confidence']:.1f}%")

    print(f"\n--- TRAJECTORY ANALYSIS ---")
    print(f"Expected Min: ${analysis['min_price']:,.2f} (at minute {analysis['min_time']})")
    print(f"Expected Max: ${analysis['max_price']:,.2f} (at minute {analysis['max_time']})")
    print(f"Expected Volatility: {analysis['volatility']:.4f}%")

    # Price trajectory summary (every 10 minutes)
    print(f"\n--- PRICE TRAJECTORY (every 10 min) ---")
    for i in range(0, len(analysis['trajectory']), 10):
        time_label = f"+{i}min" if i > 0 else "Now"
        print(f"  {time_label:>6}: ${analysis['trajectory'][i]:,.2f}")

    print("\n" + "=" * 60)


def test_on_random_timestamps(model, df, n_tests=5):
    """Test model on random historical timestamps"""
    print("\n" + "=" * 60)
    print("TESTING ON RANDOM HISTORICAL TIMESTAMPS")
    print("=" * 60)

    # Calculate features for all data
    features_df = calculate_all_features(df)
    normalized_df = rolling_zscore(features_df, window=60).fillna(0)
    feature_data = normalized_df.values.astype(np.float32)

    ohlcv_features = ['open_return', 'high_return', 'low_return', 'close_return', 'volume_change']
    ohlcv_indices = [FEATURE_NAMES.index(f) for f in ohlcv_features]

    # Pick random start points
    max_start = len(feature_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON
    if max_start <= 0:
        print("Not enough data for testing!")
        return

    np.random.seed(int(datetime.now().timestamp()) % 1000)
    start_indices = np.random.choice(max_start, size=min(n_tests, max_start), replace=False)

    correct_directions = 0

    for i, start_idx in enumerate(start_indices):
        # Get input sequence
        input_seq = feature_data[start_idx:start_idx + SEQUENCE_LENGTH]
        input_seq = input_seq[np.newaxis, :, :]

        # Get actual data for comparison
        actual_start = start_idx + SEQUENCE_LENGTH
        actual_data = feature_data[actual_start:actual_start + PREDICTION_HORIZON]

        # Current price
        current_price = df['close'].iloc[actual_start - 1]
        actual_price_60min = df['close'].iloc[actual_start + PREDICTION_HORIZON - 1]
        actual_return = (actual_price_60min - current_price) / current_price

        # Predict
        predictions = predict_60_minutes(model, input_seq)
        predicted_return = np.sum(predictions[:, 3])

        # Direction accuracy
        pred_direction = "UP" if predicted_return > 0 else "DOWN"
        actual_direction = "UP" if actual_return > 0 else "DOWN"
        correct = pred_direction == actual_direction
        if correct:
            correct_directions += 1

        timestamp = datetime.fromtimestamp(df['timestamp'].iloc[actual_start] / 1000)

        print(f"\nTest {i+1}: {timestamp}")
        print(f"  Current Price: ${current_price:,.2f}")
        print(f"  Predicted 60-min Return: {predicted_return*100:+.4f}%")
        print(f"  Actual 60-min Return:    {actual_return*100:+.4f}%")
        print(f"  Predicted Price: ${current_price * (1 + predicted_return):,.2f}")
        print(f"  Actual Price:    ${actual_price_60min:,.2f}")
        print(f"  Direction: {pred_direction} vs {actual_direction} - {'CORRECT' if correct else 'WRONG'}")

    print("\n" + "=" * 60)
    accuracy = correct_directions / len(start_indices) * 100
    print(f"DIRECTION ACCURACY: {accuracy:.1f}% ({correct_directions}/{len(start_indices)})")
    print("=" * 60)


def main():
    # Check for test mode
    test_mode = '--test' in sys.argv

    # Load model
    model = load_trained_model()
    if model is None:
        return

    # Load BTC data
    print("\nLoading BTC data...")
    df = load_all_btc_candles()

    if df.empty:
        print("No data available!")
        return

    if test_mode:
        # Test on random historical timestamps
        n_tests = 10
        for arg in sys.argv:
            if arg.startswith('--n='):
                n_tests = int(arg.split('=')[1])

        test_on_random_timestamps(model, df, n_tests=n_tests)
    else:
        # Predict from latest data
        input_seq, context = prepare_latest_sequence(df)

        if input_seq is None:
            return

        print(f"\nGenerating 60-minute prediction...")
        predictions = predict_60_minutes(model, input_seq)

        # Analyze and display
        analysis = analyze_prediction(predictions, context['current_price'])
        print_prediction(analysis, context)

        # Save prediction to file
        output = {
            'generated_at': datetime.now().isoformat(),
            'prediction_start': context['datetime'].isoformat(),
            'current_price': float(context['current_price']),
            'predicted_price': float(analysis['predicted_price']),
            'price_change_pct': float(analysis['price_change_pct']),
            'signal': analysis['signal'],
            'confidence': float(analysis['confidence']),
            'trajectory': [float(p) for p in analysis['trajectory']]
        }

        output_path = os.path.join(os.path.dirname(__file__), 'latest_prediction.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nPrediction saved to: {output_path}")


if __name__ == '__main__':
    main()
