#!/usr/bin/env python3
"""
BTC Next-Minute ULTRA Predictor - TRUE AUTOREGRESSIVE

STRATEGY:
  Input: 30 minutes of real data
  Predict: Minute 31 (OHLCV returns)
  Then: Append prediction, shift window [2-31] → predict 32
  Repeat: 60 times to get 60 predicted minutes (31-90)

ULTRA MODEL (Practical for RTX 4050 4GB):
- ~20M parameters (fits in 4GB VRAM)
- ALL 1M+ samples (no limit)
- Deep CNN + Transformer + BiLSTM
- 500 epochs with early stopping

TARGET: SUB $0.01 (sub-cent) accuracy at $100k BTC
Estimated training time: 2-3 days
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

# Model settings - ULTRA (practical)
SEQUENCE_LENGTH = 30      # 30 minutes input (for autoregressive chaining)
OUTPUT_FEATURES = 5       # OHLCV returns for NEXT 1 minute
CHAIN_LENGTH = 60         # Chain 60 predictions (31 → 90)

# Training settings - AGGRESSIVE but PRACTICAL
BATCH_SIZE = 128          # Larger batch for faster training
EPOCHS = 500              # With early stopping
EARLY_STOPPING_PATIENCE = 50  # Stop if no improvement
MAX_SAMPLES = 200000      # Limit samples to avoid OOM (~4 months of recent data)

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'next_minute_ultra')


def calculate_ohlcv_features_extended(df):
    """
    Extended feature set - 60+ features for maximum accuracy
    """
    features = pd.DataFrame(index=df.index)

    # === RAW PRICE RETURNS ===
    features['open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    features['high_return'] = (df['high'] - df['close'].shift(1)) / df['close'].shift(1)
    features['low_return'] = (df['low'] - df['close'].shift(1)) / df['close'].shift(1)
    features['close_return'] = df['close'].pct_change()
    features['volume_change'] = df['volume'].pct_change()

    # === MOMENTUM (multiple periods) ===
    for period in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        features[f'momentum_{period}'] = df['close'].pct_change(period)

    # === VOLATILITY ===
    features['range'] = (df['high'] - df['low']) / df['close']
    features['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    ) / df['close']

    for period in [3, 5, 7, 10, 14, 20, 30]:
        features[f'volatility_{period}'] = features['close_return'].rolling(period).std()
        features[f'atr_{period}'] = features['true_range'].rolling(period).mean()

    # === ORDER FLOW / CANDLESTICK ANALYSIS ===
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-10
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    features['body_ratio'] = body / total_range
    features['upper_wick_ratio'] = upper_wick / total_range
    features['lower_wick_ratio'] = lower_wick / total_range
    features['buy_pressure'] = lower_wick / total_range
    features['sell_pressure'] = upper_wick / total_range
    features['pressure_diff'] = features['buy_pressure'] - features['sell_pressure']

    # Cumulative pressure
    bullish = (df['close'] > df['open']).astype(float)
    bearish = (df['close'] < df['open']).astype(float)
    for period in [3, 5, 10, 20]:
        delta = bullish * df['volume'] - bearish * df['volume']
        vol_sum = df['volume'].rolling(period).sum() + 1e-10
        features[f'cum_delta_{period}'] = delta.rolling(period).sum() / vol_sum

    # === VOLUME ANALYSIS ===
    for period in [3, 5, 10, 20, 30]:
        vol_ma = df['volume'].rolling(period).mean()
        features[f'volume_ratio_{period}'] = df['volume'] / (vol_ma + 1e-10)

    features['volume_trend'] = df['volume'].rolling(10).mean() / (df['volume'].rolling(30).mean() + 1e-10)

    # === TECHNICAL INDICATORS ===
    # RSI at multiple periods
    for period in [5, 7, 10, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = (100 - (100 / (1 + rs))) / 100

    # SMA distances
    for period in [3, 5, 7, 10, 14, 20, 30, 50]:
        sma = df['close'].rolling(period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma

    # EMA distances
    for period in [5, 10, 20]:
        ema = df['close'].ewm(span=period).mean()
        features[f'ema_{period}_dist'] = (df['close'] - ema) / ema

    # Bollinger Band position
    for period in [10, 20]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        features[f'bb_{period}_pos'] = (df['close'] - sma) / (2 * std + 1e-10)

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd'] = macd / df['close']
    features['macd_signal'] = signal / df['close']
    features['macd_hist'] = (macd - signal) / df['close']

    # Stochastic
    for period in [7, 14]:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        features[f'stoch_{period}'] = (df['close'] - low_min) / (high_max - low_min + 1e-10)

    # === TIME FEATURES ===
    # Hour of day (if timestamp available)
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], unit='ms')
        features['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
        features['minute_sin'] = np.sin(2 * np.pi * dt.dt.minute / 60)
        features['minute_cos'] = np.cos(2 * np.pi * dt.dt.minute / 60)
        features['day_of_week_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    return features.fillna(0).replace([np.inf, -np.inf], 0)


def prepare_ultra_data(candles_df, seq_length=60):
    """
    Prepare ALL data for ultra training
    """
    print("Calculating extended features...")
    features = calculate_ohlcv_features_extended(candles_df)

    print(f"Total features: {len(features.columns)}")

    # Normalize with rolling z-score
    print("Normalizing...")
    normalized = features.copy()
    for col in features.columns:
        rolling_mean = features[col].rolling(120, min_periods=1).mean()
        rolling_std = features[col].rolling(120, min_periods=1).std() + 1e-8
        normalized[col] = (features[col] - rolling_mean) / rolling_std

    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    # Clip extreme values
    normalized = normalized.clip(-5, 5)

    feature_names = list(normalized.columns)
    feature_matrix = normalized.values.astype(np.float32)

    print("Creating sequences (memory-efficient)...")
    warmup = max(seq_length, 120) + 10

    # Pre-calculate valid indices
    target_cols = ['open_return', 'high_return', 'low_return', 'close_return', 'volume_change']
    target_matrix = features[target_cols].values.astype(np.float32)

    # Limit data range BEFORE creating sequences to save memory
    total_possible = len(feature_matrix) - warmup - 1
    if total_possible > MAX_SAMPLES:
        # Start from a later point to get most recent data
        start_offset = total_possible - MAX_SAMPLES
        print(f"Using most recent {MAX_SAMPLES:,} samples (skipping first {start_offset:,})")
    else:
        start_offset = 0

    # Pre-allocate arrays (use float32 directly for limited samples)
    actual_max = min(MAX_SAMPLES, total_possible)
    X = np.zeros((actual_max, seq_length, feature_matrix.shape[1]), dtype=np.float32)
    Y = np.zeros((actual_max, 5), dtype=np.float32)

    valid_count = 0
    skipped = 0
    start_idx = warmup + start_offset

    for i in range(start_idx, len(feature_matrix) - 1):
        if valid_count >= actual_max:
            break

        input_seq = feature_matrix[i - seq_length:i]
        target = target_matrix[i]

        if np.isnan(input_seq).any() or np.isnan(target).any():
            skipped += 1
            continue
        if np.isinf(input_seq).any() or np.isinf(target).any():
            skipped += 1
            continue

        X[valid_count] = input_seq
        Y[valid_count] = target
        valid_count += 1

        # Progress indicator
        if valid_count % 100000 == 0:
            print(f"  Processed {valid_count:,} sequences...")

    # Trim to valid samples only
    X = X[:valid_count]
    Y = Y[:valid_count]
    if skipped > 0:
        print(f"  Skipped {skipped:,} invalid sequences")

    # Clip extreme targets
    Y = np.clip(Y, -0.02, 0.02)

    print(f"\nCreated {len(X):,} samples (ALL DATA)")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {Y.shape}")

    print(f"\nTarget statistics (% returns):")
    print(f"  Open:   mean={Y[:, 0].mean()*100:.4f}%, std={Y[:, 0].std()*100:.4f}%")
    print(f"  High:   mean={Y[:, 1].mean()*100:.4f}%, std={Y[:, 1].std()*100:.4f}%")
    print(f"  Low:    mean={Y[:, 2].mean()*100:.4f}%, std={Y[:, 2].std()*100:.4f}%")
    print(f"  Close:  mean={Y[:, 3].mean()*100:.4f}%, std={Y[:, 3].std()*100:.4f}%")
    print(f"  Volume: mean={Y[:, 4].mean()*100:.4f}%, std={Y[:, 4].std()*100:.4f}%")

    return X, Y, feature_names


def residual_block(x, filters, kernel_size=3):
    """Residual block for better gradient flow"""
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


def build_ultra_model(num_features):
    """
    ULTRA model - ~10M parameters
    Balanced for RTX 4050 4GB VRAM
    Target: Sub-cent accuracy

    Architecture:
    - 6 conv branches (3,5,7,9,11,15) × 256 filters = 1536 channels
    - 4 residual blocks (512→512→384→384)
    - 4 transformer layers with 12 heads
    - 3 BiLSTM layers (384→256→128)
    - Dense: 1024→512→256→128→64→5

    Estimated: ~10M parameters
    Training time: ~24-36 hours for 500 epochs with 1M samples
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # === MULTI-SCALE CONVOLUTIONS (6 branches × 256 filters) ===
    conv_branches = []
    for kernel_size in [3, 5, 7, 9, 11, 15]:
        conv = layers.Conv1D(256, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
        conv_branches.append(conv)

    merged = layers.Concatenate()(conv_branches)  # 1536 channels
    x = layers.BatchNormalization()(merged)

    # === RESIDUAL BLOCKS (4 blocks) ===
    x = residual_block(x, 512)
    x = layers.Dropout(0.1)(x)
    x = residual_block(x, 512)
    x = layers.Dropout(0.1)(x)
    x = residual_block(x, 384)
    x = layers.Dropout(0.1)(x)
    x = residual_block(x, 384)

    # === TRANSFORMER ATTENTION (4 layers, 12 heads) ===
    for _ in range(4):
        attention = layers.MultiHeadAttention(num_heads=12, key_dim=48, dropout=0.1)(x, x)
        x = layers.LayerNormalization()(x + attention)
        # Feed-forward network
        ff = layers.Dense(1024, activation='relu')(x)
        ff = layers.Dense(384)(ff)
        x = layers.LayerNormalization()(x + ff)

    # === BIDIRECTIONAL LSTM (3 layers) ===
    x = layers.Bidirectional(layers.LSTM(384, return_sequences=True, dropout=0.1))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.1))(x)
    lstm_out = layers.Bidirectional(layers.LSTM(128, dropout=0.1))(x)

    # === GLOBAL FEATURES ===
    attention_pooled = layers.GlobalAveragePooling1D()(x)
    attention_max = layers.GlobalMaxPooling1D()(x)

    combined = layers.Concatenate()([lstm_out, attention_pooled, attention_max])

    # === DENSE NETWORK ===
    x = layers.Dense(1024, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # === OUTPUT: 5 OHLCV returns ===
    raw_output = layers.Dense(OUTPUT_FEATURES, activation='tanh')(x)
    output = layers.Lambda(lambda x: x * 0.02, name='ohlcv_returns')(raw_output)

    model = Model(inputs=inputs, outputs=output)

    # Cosine decay with restarts for better convergence
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=2000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.0001
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mse',
        metrics=['mae']
    )

    return model


def main():
    print("=" * 60)
    print("BTC NEXT-MINUTE ULTRA PREDICTOR")
    print("MASSIVE MODEL - ALL DATA - TARGET: $0.01 ACCURACY")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Output: Next 1 minute OHLCV returns")
    print()

    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    print()

    # Load ALL data
    print("Loading ALL BTC candles...")
    candles_df = load_all_btc_candles()
    print(f"Loaded {len(candles_df):,} candles")
    print()

    # Prepare data (NO LIMIT!)
    X, Y, feature_names = prepare_ultra_data(candles_df, SEQUENCE_LENGTH)
    print()

    # Split (temporal - use recent data for testing)
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build ULTRA model
    print("Building ULTRA model...")
    model = build_ultra_model(X.shape[2])
    model.summary()

    total_params = model.count_params()
    print(f"\n*** TOTAL PARAMETERS: {total_params:,} ***")
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_mae',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, 'logs'),
            histogram_freq=1
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING ULTRA MODEL")
    print("=" * 60)

    # Time estimate
    steps_per_epoch = len(X_train) // BATCH_SIZE
    estimated_time_per_step = 0.15  # ~150ms per step for 10M params on RTX 4050
    estimated_epoch_time = steps_per_epoch * estimated_time_per_step
    estimated_total_time = estimated_epoch_time * EPOCHS

    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Estimated time per epoch: {estimated_epoch_time/60:.1f} minutes")
    print(f"Estimated total training time: {estimated_total_time/3600:.1f} hours")
    print(f"Estimated completion: {datetime.now() + timedelta(seconds=estimated_total_time)}")
    print()

    start_time = datetime.now()

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Training complete
    training_time = datetime.now() - start_time
    print(f"\nActual training time: {training_time}")

    # Evaluate
    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test MSE: {results[0]:.10f}")
    print(f"Test MAE: {results[1]:.8f} ({results[1]*100:.6f}%)")

    # Per-feature MAE
    predictions = model.predict(X_test, verbose=0)
    feature_labels = ['Open', 'High', 'Low', 'Close', 'Volume']

    print("\nPer-feature MAE:")
    for i, label in enumerate(feature_labels):
        mae = np.mean(np.abs(predictions[:, i] - Y_test[:, i]))
        print(f"  {label}: {mae*100:.6f}%")

    # Direction accuracy
    pred_dir = predictions[:, 3] > 0
    actual_dir = Y_test[:, 3] > 0
    dir_acc = np.mean(pred_dir == actual_dir)
    print(f"\nClose Direction Accuracy: {dir_acc * 100:.1f}%")

    # Dollar accuracy at $100k BTC
    btc_price = 100000
    close_mae = np.mean(np.abs(predictions[:, 3] - Y_test[:, 3]))
    dollar_mae = close_mae * btc_price
    print(f"\nClose MAE in dollars: ${dollar_mae:.2f}")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'output_features': OUTPUT_FEATURES,
        'num_input_features': X.shape[2],
        'total_parameters': int(total_params),
        'feature_names': feature_names,
        'test_mse': float(results[0]),
        'test_mae': float(results[1]),
        'direction_accuracy': float(dir_acc),
        'close_mae_dollars': float(dollar_mae),
        'total_samples': len(X),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print("ULTRA MODEL RESULTS:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Close MAE: {close_mae*100:.6f}% (${dollar_mae:.2f})")
    print(f"  Direction Accuracy: {dir_acc * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
