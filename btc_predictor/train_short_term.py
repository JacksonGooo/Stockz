#!/usr/bin/env python3
"""
BTC Short-Term Predictor (15 minutes)
Shorter prediction horizon = potentially higher accuracy
Uses order flow approximations + technical features
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES
from data_loader import load_all_btc_candles

# Model settings - SHORT TERM
SEQUENCE_LENGTH = 30      # Use 30 minutes of input data
PREDICTION_HORIZON = 15   # Predict only 15 minutes ahead (shorter = more accurate)

# Training settings
BATCH_SIZE = 256
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
MAX_SAMPLES = 500000

# Output directory - separate from 60-min model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'short_term_15min')


def calculate_order_flow_features(df):
    """Calculate order flow approximations from OHLCV"""
    features = pd.DataFrame(index=df.index)

    # Buy/Sell pressure from candle structure
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-10

    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    features['buy_pressure'] = lower_wick / total_range
    features['sell_pressure'] = upper_wick / total_range
    features['pressure_imbalance'] = features['buy_pressure'] - features['sell_pressure']

    # Volume analysis
    features['volume_buy_pressure'] = features['buy_pressure'] * df['volume']
    features['volume_sell_pressure'] = features['sell_pressure'] * df['volume']

    # Cumulative delta approximation
    bullish = (df['close'] > df['open']).astype(float)
    bearish = (df['close'] < df['open']).astype(float)
    features['cumulative_delta'] = (bullish * df['volume'] - bearish * df['volume']).rolling(10).sum()
    features['cumulative_delta_norm'] = features['cumulative_delta'] / (df['volume'].rolling(10).sum() + 1e-10)

    # Volume spikes
    vol_sma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std() + 1e-10
    features['volume_zscore'] = (df['volume'] - vol_sma) / vol_std
    features['large_trade_count'] = (features['volume_zscore'] > 2).rolling(5).sum()

    # Momentum with volume
    price_change = df['close'].pct_change()
    features['momentum_volume'] = price_change * features['volume_zscore']

    # Short-term momentum indicators
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_10'] = df['close'].pct_change(10)
    features['acceleration'] = features['momentum_5'] - features['momentum_10']

    return features


def prepare_short_term_data(candles_df, sequence_length=30, horizon=15):
    """Prepare data for 15-minute prediction"""
    print("Calculating standard features...")
    features = calculate_all_features(candles_df)

    print("Calculating order flow features...")
    order_flow = calculate_order_flow_features(candles_df)

    # Combine
    all_features = pd.concat([features, order_flow], axis=1)

    print("Normalizing...")
    normalized = rolling_zscore(all_features, window=60)
    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    feature_names = list(normalized.columns)
    feature_matrix = normalized.values.astype(np.float32)

    print(f"Total features: {len(feature_names)}")

    close_prices = candles_df['close'].values

    X = []
    y_direction = []
    y_change = []

    print("Creating sequences...")

    for i in range(sequence_length + 60, len(feature_matrix) - horizon):  # +60 for warmup
        seq = feature_matrix[i - sequence_length:i]

        if np.isnan(seq).any() or np.isinf(seq).any():
            continue

        current_price = close_prices[i - 1]
        future_price = close_prices[i + horizon - 1]

        percent_change = (future_price - current_price) / current_price * 100
        direction = 1 if percent_change > 0 else 0

        X.append(seq)
        y_direction.append(direction)
        y_change.append(np.clip(percent_change, -5, 5))

    X = np.array(X, dtype=np.float32)
    y_direction = np.array(y_direction, dtype=np.float32)
    y_change = np.array(y_change, dtype=np.float32)

    up_count = np.sum(y_direction == 1)
    down_count = np.sum(y_direction == 0)
    print(f"Samples: {len(X):,}")
    print(f"Class balance: UP={up_count} ({up_count/len(y_direction)*100:.1f}%), DOWN={down_count}")
    print(f"Price change: mean={np.mean(y_change):.4f}%, std={np.std(y_change):.4f}%")

    return X, y_direction, y_change, feature_names


def build_model(num_features):
    """Build model optimized for short-term prediction"""
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # Attention mechanism for short-term patterns
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = layers.LayerNormalization()(x + inputs)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.2)(x)

    # Dense
    shared = layers.Dense(32, activation='relu')(x)

    # Direction output
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(shared)

    # Price change output
    price_output = layers.Dense(1, activation='linear', name='price_change')(shared)

    model = Model(inputs=inputs, outputs=[direction_output, price_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={'direction': 'binary_crossentropy', 'price_change': 'mse'},
        loss_weights={'direction': 1.0, 'price_change': 0.05},
        metrics={'direction': 'accuracy', 'price_change': 'mae'}
    )

    return model


def main():
    print("=" * 60)
    print("BTC SHORT-TERM PREDICTOR (15 MIN)")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Predict: {PREDICTION_HORIZON} minutes ahead")
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

    # Prepare
    X, y_dir, y_chg, feature_names = prepare_short_term_data(
        candles_df, SEQUENCE_LENGTH, PREDICTION_HORIZON
    )
    print()

    # Limit samples
    if len(X) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X, y_dir, y_chg = X[idx], y_dir[idx], y_chg[idx]

    # Split
    X_train, X_test, y_dir_train, y_dir_test, y_chg_train, y_chg_test = train_test_split(
        X, y_dir, y_chg, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build
    print("Building model...")
    model = build_model(X.shape[2])
    model.summary()
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_direction_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_direction_accuracy',
            save_best_only=True, mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train, {'direction': y_dir_train, 'price_change': y_chg_train},
        validation_data=(X_test, {'direction': y_dir_test, 'price_change': y_chg_test}),
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
        X_test, {'direction': y_dir_test, 'price_change': y_chg_test}, verbose=0
    )

    print(f"Direction Accuracy: {results[3] * 100:.1f}%")
    print(f"Price MAE: {results[4]:.4f}%")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': X.shape[2],
        'feature_names': feature_names,
        'direction_accuracy': float(results[3]),
        'price_mae': float(results[4]),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print(f"15-MIN DIRECTION ACCURACY: {results[3] * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
