"""
Data Loader for BTC Autoregressive Prediction
Loads and preprocesses 1-minute BTC candle data for training
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from features import calculate_all_features, rolling_zscore, NUM_FEATURES, FEATURE_NAMES


# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Crypto', 'BTC')
SEQUENCE_LENGTH = 30  # 30-minute lookback (user requested)
PREDICTION_HORIZON = 60  # Predict 60 minutes ahead


def load_all_btc_candles():
    """Load all BTC candle data from weekly folders"""
    all_candles = []

    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return pd.DataFrame()

    # Iterate through week folders
    week_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if d.startswith('week_') and os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    for week_folder in week_folders:
        week_path = os.path.join(DATA_DIR, week_folder)
        day_files = sorted([f for f in os.listdir(week_path) if f.endswith('.json')])

        for day_file in day_files:
            file_path = os.path.join(week_path, day_file)
            try:
                with open(file_path, 'r') as f:
                    candles = json.load(f)
                    all_candles.extend(candles)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not all_candles:
        print("No candles loaded!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_candles)

    # Ensure columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return pd.DataFrame()

    # Sort by timestamp and remove duplicates
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

    print(f"Loaded {len(df):,} candles from {len(week_folders)} weeks")
    print(f"Date range: {datetime.fromtimestamp(df['timestamp'].iloc[0]/1000)} to {datetime.fromtimestamp(df['timestamp'].iloc[-1]/1000)}")

    return df


def prepare_training_data(df, sequence_length=SEQUENCE_LENGTH, prediction_horizon=PREDICTION_HORIZON):
    """
    Prepare sequences for training the autoregressive model

    Args:
        df: DataFrame with OHLCV data
        sequence_length: Number of candles to use as input (30)
        prediction_horizon: Number of candles to predict (60, but we train on next-candle)

    Returns:
        X: Input sequences (N, sequence_length, num_features)
        y: Target values (N, 5) - next candle's OHLCV returns
    """
    print("Calculating features...")
    features_df = calculate_all_features(df)

    print("Applying rolling z-score normalization...")
    normalized_df = rolling_zscore(features_df, window=60)

    # Fill NaN values (from feature calculation warmup period)
    normalized_df = normalized_df.fillna(0)

    # Convert to numpy
    feature_data = normalized_df.values.astype(np.float32)

    # Create sequences
    print("Creating training sequences...")
    X = []
    y = []

    # Target columns indices (OHLCV returns)
    target_cols = ['open_return', 'high_return', 'low_return', 'close_return', 'volume_change']
    target_indices = [FEATURE_NAMES.index(col) for col in target_cols]

    # We need sequence_length for input + 1 for target
    for i in range(len(feature_data) - sequence_length - 1):
        # Input: sequence_length candles
        X.append(feature_data[i:i + sequence_length])

        # Target: next candle's OHLCV returns
        y.append(feature_data[i + sequence_length, target_indices])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Clip extreme values in targets
    y = np.clip(y, -0.1, 0.1)  # Clip to +/- 10% returns

    print(f"Created {len(X):,} training sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


def train_test_split_temporal(X, y, test_ratio=0.2):
    """
    Split data temporally (not randomly) to avoid lookahead bias
    """
    split_idx = int(len(X) * (1 - test_ratio))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    return X_train, y_train, X_test, y_test


def get_random_test_samples(df, n_samples=10, sequence_length=SEQUENCE_LENGTH, prediction_horizon=PREDICTION_HORIZON):
    """
    Get random samples for testing predictions against reality

    Returns list of (input_sequence, actual_60min_candles, start_timestamp)
    """
    # Calculate features
    features_df = calculate_all_features(df)
    normalized_df = rolling_zscore(features_df, window=60).fillna(0)
    feature_data = normalized_df.values.astype(np.float32)

    # We need at least sequence_length + prediction_horizon candles
    max_start = len(feature_data) - sequence_length - prediction_horizon

    if max_start <= 0:
        print("Not enough data for testing!")
        return []

    # Pick random start indices
    np.random.seed(42)  # For reproducibility
    start_indices = np.random.choice(max_start, size=min(n_samples, max_start), replace=False)

    samples = []
    for start_idx in start_indices:
        input_seq = feature_data[start_idx:start_idx + sequence_length]
        actual_candles = feature_data[start_idx + sequence_length:start_idx + sequence_length + prediction_horizon]
        timestamp = df['timestamp'].iloc[start_idx + sequence_length]

        samples.append({
            'input_sequence': input_seq,
            'actual_candles': actual_candles,
            'start_timestamp': timestamp,
            'start_datetime': datetime.fromtimestamp(timestamp / 1000)
        })

    return samples


def load_and_prepare_data():
    """Main function to load and prepare all training data"""
    # Load raw candles
    df = load_all_btc_candles()

    if df.empty:
        return None, None, None, None, None

    # Prepare training data
    X, y = prepare_training_data(df)

    # Split temporally
    X_train, y_train, X_test, y_test = train_test_split_temporal(X, y)

    return X_train, y_train, X_test, y_test, df


if __name__ == '__main__':
    print("Testing data loader...")
    print("=" * 60)

    # Load all data
    df = load_all_btc_candles()

    if not df.empty:
        print("\n" + "=" * 60)
        print("Preparing training data...")
        X, y = prepare_training_data(df)

        print("\n" + "=" * 60)
        print("Getting random test samples...")
        samples = get_random_test_samples(df, n_samples=5)
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}: {sample['start_datetime']}")

        print("\n" + "=" * 60)
        print("Data loading test complete!")
