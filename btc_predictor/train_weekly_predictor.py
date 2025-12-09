#!/usr/bin/env python3
"""
BTC Weekly Predictor
Predicts next week's HIGH, LOW, MEAN from daily candle data

Input: Last 30 days of daily candles
Output: Next week's predicted high, low, and mean price

This is potentially more accurate because:
1. Daily data has less noise than minute data
2. Weekly trends follow macro patterns
3. Longer timeframe = easier prediction
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model

from data_loader import load_all_btc_candles

# Model settings
DAYS_INPUT = 30      # Use 30 days of daily data
DAYS_PREDICT = 7     # Predict next 7 days (1 week)

# Training settings
BATCH_SIZE = 64
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 30

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'weekly_predictor')


def aggregate_to_daily(minute_df):
    """Convert minute candles to daily candles"""
    minute_df = minute_df.copy()
    minute_df['date'] = pd.to_datetime(minute_df['timestamp'], unit='ms').dt.date

    daily = minute_df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'timestamp': 'first'
    }).reset_index()

    return daily


def calculate_daily_features(daily_df):
    """Calculate features from daily candles"""
    df = daily_df.copy()

    # Price returns
    df['daily_return'] = df['close'].pct_change()
    df['high_return'] = (df['high'] - df['close'].shift(1)) / df['close'].shift(1)
    df['low_return'] = (df['low'] - df['close'].shift(1)) / df['close'].shift(1)

    # Volatility
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_7d'] = df['daily_return'].rolling(7).std()
    df['volatility_14d'] = df['daily_return'].rolling(14).std()

    # Moving averages
    for period in [7, 14, 21, 30]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / df['close']

    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_sma_7'] = df['volume'].rolling(7).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_7']

    # Weekly patterns
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek / 6  # 0-1

    return df


def prepare_weekly_prediction_data(daily_df, days_input=30, days_predict=7):
    """
    Prepare data for weekly prediction

    Input: 30 days of daily features
    Output: Next week's high, low, mean (as % change from last close)
    """
    df = calculate_daily_features(daily_df)

    # Drop rows with NaN
    df = df.dropna()

    # Feature columns
    feature_cols = [
        'daily_return', 'high_return', 'low_return', 'daily_range',
        'volatility_7d', 'volatility_14d',
        'sma_7_dist', 'sma_14_dist', 'sma_21_dist', 'sma_30_dist',
        'rsi', 'macd', 'volume_change', 'volume_ratio', 'day_of_week'
    ]

    # Normalize features with rolling z-score
    feature_df = df[feature_cols].copy()
    for col in feature_cols:
        if col != 'day_of_week':
            rolling_mean = feature_df[col].rolling(30, min_periods=1).mean()
            rolling_std = feature_df[col].rolling(30, min_periods=1).std() + 1e-8
            feature_df[col] = (feature_df[col] - rolling_mean) / rolling_std

    feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)
    feature_matrix = feature_df.values.astype(np.float32)

    # Get close prices for target calculation
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    X = []
    Y = []

    # Create samples
    for i in range(days_input, len(feature_matrix) - days_predict):
        # Input: last 30 days
        input_seq = feature_matrix[i - days_input:i]

        # Current price (last close)
        current_price = closes[i - 1]

        # Next week's prices
        next_week_highs = highs[i:i + days_predict]
        next_week_lows = lows[i:i + days_predict]
        next_week_closes = closes[i:i + days_predict]

        week_high = np.max(next_week_highs)
        week_low = np.min(next_week_lows)
        week_mean = np.mean(next_week_closes)

        # Convert to % change from current price
        high_pct = (week_high - current_price) / current_price * 100
        low_pct = (week_low - current_price) / current_price * 100
        mean_pct = (week_mean - current_price) / current_price * 100

        X.append(input_seq)
        Y.append([high_pct, low_pct, mean_pct])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Clip extreme targets
    Y = np.clip(Y, -30, 30)  # ±30% max weekly move

    print(f"Created {len(X):,} weekly samples")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {Y.shape}")
    print(f"Target stats:")
    print(f"  High: mean={Y[:, 0].mean():.2f}%, std={Y[:, 0].std():.2f}%")
    print(f"  Low:  mean={Y[:, 1].mean():.2f}%, std={Y[:, 1].std():.2f}%")
    print(f"  Mean: mean={Y[:, 2].mean():.2f}%, std={Y[:, 2].std():.2f}%")

    return X, Y, feature_cols


def build_weekly_model(num_features):
    """Build model for weekly high/low/mean prediction"""
    inputs = layers.Input(shape=(DAYS_INPUT, num_features))

    # Transformer encoder
    x = layers.Dense(64)(inputs)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization()(x + attention)

    # LSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.2)(x)

    # Dense
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Output: high, low, mean (3 values)
    output = layers.Dense(3, name='weekly_targets')(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def main():
    print("=" * 60)
    print("BTC WEEKLY PREDICTOR")
    print("Predicting Weekly HIGH, LOW, MEAN from Daily Data")
    print("=" * 60)
    print(f"Input: {DAYS_INPUT} days")
    print(f"Predict: {DAYS_PREDICT} days ahead (1 week)")
    print()

    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    print()

    # Load minute data
    print("Loading BTC minute candles...")
    minute_df = load_all_btc_candles()
    print(f"Loaded {len(minute_df):,} minute candles")

    # Convert to daily
    print("\nConverting to daily candles...")
    daily_df = aggregate_to_daily(minute_df)
    print(f"Created {len(daily_df):,} daily candles")
    print(f"Date range: {daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")
    print()

    # Prepare data
    X, Y, feature_cols = prepare_weekly_prediction_data(daily_df, DAYS_INPUT, DAYS_PREDICT)
    print()

    # Split temporally
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build model
    print("Building weekly predictor model...")
    model = build_weekly_model(len(feature_cols))
    model.summary()
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_mae', save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
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

    results = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss (MSE): {results[0]:.4f}")
    print(f"Test MAE: {results[1]:.2f}%")

    # Detailed analysis
    predictions = model.predict(X_test, verbose=0)

    print("\nPer-target MAE:")
    print(f"  High prediction MAE:  {np.mean(np.abs(predictions[:, 0] - Y_test[:, 0])):.2f}%")
    print(f"  Low prediction MAE:   {np.mean(np.abs(predictions[:, 1] - Y_test[:, 1])):.2f}%")
    print(f"  Mean prediction MAE:  {np.mean(np.abs(predictions[:, 2] - Y_test[:, 2])):.2f}%")

    # Direction accuracy for weekly mean
    pred_direction = predictions[:, 2] > 0
    actual_direction = Y_test[:, 2] > 0
    direction_acc = np.mean(pred_direction == actual_direction)
    print(f"\nWeekly direction accuracy: {direction_acc * 100:.1f}%")

    # Save config
    config = {
        'days_input': DAYS_INPUT,
        'days_predict': DAYS_PREDICT,
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'test_mae': float(results[1]),
        'direction_accuracy': float(direction_acc),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print(f"WEEKLY PREDICTION MAE: {results[1]:.2f}%")
    print(f"WEEKLY DIRECTION ACCURACY: {direction_acc * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
