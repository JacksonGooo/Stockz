#!/usr/bin/env python3
"""
BTC 1-Minute Ahead Predictor
Krafer-style: Predict the NEXT candle with maximum precision

Key insight: 1-minute predictions are much more feasible because:
1. Order flow has strong predictive power at short timeframes
2. Market microstructure patterns are more reliable
3. Less time for external events to disrupt

Target: Predict next candle's CLOSE price with cent-level accuracy

Architecture: ~2M parameters (like Krafer claims)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model

from data_loader import load_all_btc_candles

# Model settings
SEQUENCE_LENGTH = 60      # Use 60 minutes of history
PREDICTION_AHEAD = 1      # Predict just 1 minute ahead

# Training settings
BATCH_SIZE = 256
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
MAX_SAMPLES = 500000

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', '1min_predictor')


def calculate_microstructure_features(df):
    """
    Calculate features focused on market microstructure
    These are most predictive for short-term price movements
    """
    features = pd.DataFrame(index=df.index)

    # === PRICE MOMENTUM (multiple scales) ===
    for period in [1, 2, 3, 5, 10, 15, 30, 60]:
        features[f'return_{period}'] = df['close'].pct_change(period)

    # === VOLATILITY ===
    features['range'] = (df['high'] - df['low']) / df['close']
    features['range_ma5'] = features['range'].rolling(5).mean()
    features['range_ma20'] = features['range'].rolling(20).mean()

    for period in [5, 10, 20, 30]:
        features[f'volatility_{period}'] = features['return_1'].rolling(period).std()

    # === ORDER FLOW APPROXIMATION ===
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-10
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    features['body_ratio'] = body / total_range
    features['upper_wick_ratio'] = upper_wick / total_range
    features['lower_wick_ratio'] = lower_wick / total_range

    # Buy/sell pressure
    features['buy_pressure'] = lower_wick / total_range
    features['sell_pressure'] = upper_wick / total_range
    features['pressure_imbalance'] = features['buy_pressure'] - features['sell_pressure']

    # Rolling pressure
    for period in [3, 5, 10, 20]:
        features[f'pressure_ma{period}'] = features['pressure_imbalance'].rolling(period).mean()

    # === CUMULATIVE DELTA (volume direction) ===
    bullish = (df['close'] > df['open']).astype(float)
    bearish = (df['close'] < df['open']).astype(float)

    for period in [5, 10, 20, 30]:
        delta = bullish * df['volume'] - bearish * df['volume']
        vol_sum = df['volume'].rolling(period).sum() + 1e-10
        features[f'cum_delta_{period}'] = delta.rolling(period).sum() / vol_sum

    # === VOLUME ANALYSIS ===
    features['volume_change'] = df['volume'].pct_change()
    for period in [5, 10, 20]:
        vol_ma = df['volume'].rolling(period).mean()
        features[f'volume_ratio_{period}'] = df['volume'] / (vol_ma + 1e-10)

    # Volume spike detection
    vol_std = df['volume'].rolling(20).std() + 1e-10
    vol_mean = df['volume'].rolling(20).mean()
    features['volume_zscore'] = (df['volume'] - vol_mean) / vol_std

    # === TECHNICAL INDICATORS ===
    # RSI (short-term)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / (loss + 1e-10)
    features['rsi_7'] = 100 - (100 / (1 + rs))

    gain14 = delta.where(delta > 0, 0).rolling(14).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs14 = gain14 / (loss14 + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs14))

    # Stochastic
    for period in [7, 14, 21]:
        lowest = df['low'].rolling(period).min()
        highest = df['high'].rolling(period).max()
        features[f'stoch_{period}'] = (df['close'] - lowest) / (highest - lowest + 1e-10)

    # MACD components
    ema5 = df['close'].ewm(span=5).mean()
    ema10 = df['close'].ewm(span=10).mean()
    ema20 = df['close'].ewm(span=20).mean()
    features['macd_fast'] = (ema5 - ema10) / df['close']
    features['macd_slow'] = (ema10 - ema20) / df['close']

    # === MOVING AVERAGE DISTANCES ===
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        ema = df['close'].ewm(span=period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma
        features[f'ema_{period}_dist'] = (df['close'] - ema) / ema

    # === PRICE LEVELS ===
    features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Distance from recent high/low
    features['dist_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
    features['dist_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

    # === TIME FEATURES ===
    df_time = pd.to_datetime(df['timestamp'], unit='ms')
    features['hour_sin'] = np.sin(2 * np.pi * df_time.dt.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * df_time.dt.hour / 24)
    features['minute_sin'] = np.sin(2 * np.pi * df_time.dt.minute / 60)
    features['minute_cos'] = np.cos(2 * np.pi * df_time.dt.minute / 60)

    return features.fillna(0).replace([np.inf, -np.inf], 0)


def prepare_1min_data(candles_df, seq_length=60):
    """
    Prepare data for 1-minute ahead prediction

    Target: Predict the EXACT price change in the next minute
    """
    print("Calculating microstructure features...")
    features = calculate_microstructure_features(candles_df)

    print(f"Total features: {len(features.columns)}")

    # Normalize with rolling z-score
    print("Normalizing features...")
    normalized = features.copy()
    for col in features.columns:
        if 'sin' not in col and 'cos' not in col:  # Don't normalize cyclic features
            rolling_mean = features[col].rolling(60, min_periods=1).mean()
            rolling_std = features[col].rolling(60, min_periods=1).std() + 1e-8
            normalized[col] = (features[col] - rolling_mean) / rolling_std

    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    feature_names = list(normalized.columns)
    feature_matrix = normalized.values.astype(np.float32)

    # Get prices for target calculation
    close_prices = candles_df['close'].values
    high_prices = candles_df['high'].values
    low_prices = candles_df['low'].values
    open_prices = candles_df['open'].values

    X = []
    y_close_change = []    # % change in close
    y_direction = []       # UP or DOWN
    y_high_change = []     # High of next candle
    y_low_change = []      # Low of next candle

    print("Creating sequences...")
    warmup = max(seq_length, 60) + 10

    for i in range(warmup, len(feature_matrix) - PREDICTION_AHEAD):
        seq = feature_matrix[i - seq_length:i]

        if np.isnan(seq).any() or np.isinf(seq).any():
            continue

        current_close = close_prices[i - 1]
        next_close = close_prices[i]
        next_high = high_prices[i]
        next_low = low_prices[i]

        # Calculate targets as % change from current close
        close_change = (next_close - current_close) / current_close * 100
        high_change = (next_high - current_close) / current_close * 100
        low_change = (next_low - current_close) / current_close * 100
        direction = 1 if close_change > 0 else 0

        X.append(seq)
        y_close_change.append(np.clip(close_change, -2, 2))  # Clip to ±2%
        y_direction.append(direction)
        y_high_change.append(np.clip(high_change, 0, 2))
        y_low_change.append(np.clip(low_change, -2, 0))

    X = np.array(X, dtype=np.float32)
    y_close_change = np.array(y_close_change, dtype=np.float32)
    y_direction = np.array(y_direction, dtype=np.float32)
    y_high_change = np.array(y_high_change, dtype=np.float32)
    y_low_change = np.array(y_low_change, dtype=np.float32)

    print(f"\nCreated {len(X):,} samples")
    print(f"Features: {X.shape[2]}")
    print(f"Direction balance: UP={np.mean(y_direction)*100:.1f}%")
    print(f"Close change: mean={np.mean(y_close_change)*100:.4f}%, std={np.std(y_close_change)*100:.4f}%")
    print(f"  Range: {np.min(y_close_change)*100:.4f}% to {np.max(y_close_change)*100:.4f}%")

    return X, y_close_change, y_direction, y_high_change, y_low_change, feature_names


def build_large_model(num_features):
    """
    Build ~2M parameter model for maximum accuracy
    Similar scale to what Krafer claims
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # === MULTI-SCALE CONVOLUTIONS ===
    conv1 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)

    conv2 = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(inputs)
    conv2 = layers.BatchNormalization()(conv2)

    conv3 = layers.Conv1D(128, kernel_size=7, padding='same', activation='relu')(inputs)
    conv3 = layers.BatchNormalization()(conv3)

    # Merge multi-scale
    merged = layers.Concatenate()([conv1, conv2, conv3])  # (batch, 60, 384)

    # Second conv block
    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # === TRANSFORMER ATTENTION ===
    attention1 = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + attention1)

    # Feed-forward
    ff = layers.Dense(512, activation='gelu')(x)
    ff = layers.Dropout(0.2)(ff)
    ff = layers.Dense(256)(ff)
    x = layers.LayerNormalization()(x + ff)

    # Second attention block
    attention2 = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + attention2)

    # === BIDIRECTIONAL LSTM ===
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    lstm_out = layers.Bidirectional(layers.LSTM(64))(x)

    # === GLOBAL FEATURES ===
    global_avg = layers.GlobalAveragePooling1D()(attention2)
    global_max = layers.GlobalMaxPooling1D()(attention2)

    # Combine all
    combined = layers.Concatenate()([lstm_out, global_avg, global_max])

    # === DENSE LAYERS ===
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)

    # === OUTPUT HEADS ===
    # Direction (classification)
    dir_head = layers.Dense(64, activation='relu')(x)
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(dir_head)

    # Close price change (regression)
    close_head = layers.Dense(64, activation='relu')(x)
    close_output = layers.Dense(1, activation='linear', name='close_change')(close_head)

    # High bound
    high_head = layers.Dense(64, activation='relu')(x)
    high_output = layers.Dense(1, activation='relu', name='high_change')(high_head)

    # Low bound
    low_head = layers.Dense(64, activation='relu')(x)
    low_output = layers.Dense(1, activation='linear', name='low_change')(low_head)

    model = Model(
        inputs=inputs,
        outputs=[direction_output, close_output, high_output, low_output]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'direction': 'binary_crossentropy',
            'close_change': 'mse',
            'high_change': 'mse',
            'low_change': 'mse'
        },
        loss_weights={
            'direction': 1.0,
            'close_change': 1.0,
            'high_change': 0.5,
            'low_change': 0.5
        },
        metrics={
            'direction': 'accuracy',
            'close_change': 'mae'
        }
    )

    return model


def main():
    print("=" * 60)
    print("BTC 1-MINUTE AHEAD PREDICTOR")
    print("Krafer-Style: Maximum Precision for Next Candle")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes of history")
    print(f"Predict: Next 1 minute (OHLC)")
    print()

    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    print()

    # Load data
    print("Loading BTC candles...")
    candles_df = load_all_btc_candles()
    print(f"Loaded {len(candles_df):,} candles")
    print()

    # Prepare data
    X, y_close, y_dir, y_high, y_low, feature_names = prepare_1min_data(
        candles_df, SEQUENCE_LENGTH
    )
    print()

    # Limit samples
    if len(X) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[idx]
        y_close = y_close[idx]
        y_dir = y_dir[idx]
        y_high = y_high[idx]
        y_low = y_low[idx]

    # Split (temporal)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_close_train, y_close_test = y_close[:split_idx], y_close[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
    y_high_train, y_high_test = y_high[:split_idx], y_high[split_idx:]
    y_low_train, y_low_test = y_low[:split_idx], y_low[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build model
    print("Building large model (~2M parameters)...")
    model = build_large_model(X.shape[2])
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_close_change_mae',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_close_change_mae',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        {
            'direction': y_dir_train,
            'close_change': y_close_train,
            'high_change': y_high_train,
            'low_change': y_low_train
        },
        validation_data=(
            X_test,
            {
                'direction': y_dir_test,
                'close_change': y_close_test,
                'high_change': y_high_test,
                'low_change': y_low_test
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = model.evaluate(
        X_test,
        {
            'direction': y_dir_test,
            'close_change': y_close_test,
            'high_change': y_high_test,
            'low_change': y_low_test
        },
        verbose=0
    )

    # Parse results
    dir_acc = results[5]  # direction_accuracy
    close_mae = results[6]  # close_change_mae

    print(f"Direction Accuracy: {dir_acc * 100:.1f}%")
    print(f"Close Change MAE: {close_mae:.6f}% ({close_mae * 100:.4f} basis points)")

    # Calculate dollar accuracy (at $100k BTC)
    btc_price = 100000
    mae_dollars = close_mae / 100 * btc_price
    print(f"\nPrice MAE in dollars (at $100k BTC): ${mae_dollars:.2f}")
    print(f"Price MAE in cents: {mae_dollars * 100:.1f} cents")

    # Detailed analysis
    predictions = model.predict(X_test, verbose=0)
    y_dir_pred = (predictions[0] > 0.5).astype(int).flatten()
    y_close_pred = predictions[1].flatten()

    # Direction accuracy when confident
    dir_probs = predictions[0].flatten()
    high_conf_mask = (dir_probs > 0.7) | (dir_probs < 0.3)
    if high_conf_mask.sum() > 0:
        hc_acc = np.mean(y_dir_pred[high_conf_mask] == y_dir_test[high_conf_mask])
        print(f"\nHigh Confidence Direction Accuracy: {hc_acc * 100:.1f}%")
        print(f"  ({high_conf_mask.sum()} / {len(y_dir_test)} predictions)")

    # Price prediction correlation
    correlation = np.corrcoef(y_close_pred, y_close_test)[0, 1]
    print(f"\nPrice Prediction Correlation: {correlation:.4f}")

    # Show sample predictions
    print("\nSample Predictions (first 10 test samples):")
    print("-" * 50)
    for i in range(min(10, len(y_close_pred))):
        pred_change = y_close_pred[i] / 100 * btc_price
        actual_change = y_close_test[i] / 100 * btc_price
        error = abs(pred_change - actual_change)
        print(f"  Pred: ${pred_change:+.2f}, Actual: ${actual_change:+.2f}, Error: ${error:.2f}")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_ahead': PREDICTION_AHEAD,
        'num_features': X.shape[2],
        'feature_names': feature_names,
        'total_params': total_params,
        'direction_accuracy': float(dir_acc),
        'close_mae_percent': float(close_mae),
        'close_mae_dollars': float(mae_dollars),
        'price_correlation': float(correlation),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print("1-MINUTE PREDICTOR RESULTS:")
    print(f"  Direction Accuracy: {dir_acc * 100:.1f}%")
    print(f"  Price MAE: ${mae_dollars:.2f} ({mae_dollars * 100:.1f} cents)")
    print(f"  Correlation: {correlation:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
