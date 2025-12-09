#!/usr/bin/env python3
"""
BTC Multi-Timeframe Predictor
Based on arxiv paper: "Neural Network-Based Algorithmic Trading Systems"

Key architecture:
- Multi-head CNN with soft attention mechanism
- Three specialized CNN heads for different timeframes (1-min, 5-min, 15-min)
- Soft attention dynamically weights outputs based on market conditions
- ~520K parameters for efficiency

This approach leverages bidirectional information flow:
- Longer timeframes provide directional bias
- Shorter timeframes reveal early signals
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

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES
from data_loader import load_all_btc_candles

# Model settings
SEQUENCE_1MIN = 30    # 30 x 1-minute candles
SEQUENCE_5MIN = 30    # 30 x 5-minute candles (150 minutes)
SEQUENCE_15MIN = 30   # 30 x 15-minute candles (450 minutes)
PREDICTION_HORIZON = 60  # Predict 60 minutes ahead

# Training settings
BATCH_SIZE = 128
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
MAX_SAMPLES = 300000

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'multitimeframe')


def aggregate_to_timeframe(minute_df, period=5):
    """Aggregate minute candles to larger timeframe"""
    df = minute_df.copy()
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp_dt')

    agg_df = df.resample(f'{period}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'timestamp': 'first'
    }).dropna()

    return agg_df.reset_index(drop=True)


def calculate_features_for_timeframe(df):
    """Calculate features for a given timeframe"""
    features = pd.DataFrame(index=df.index)

    # Price returns
    features['close_return'] = df['close'].pct_change()
    features['high_return'] = (df['high'] - df['close'].shift(1)) / df['close'].shift(1)
    features['low_return'] = (df['low'] - df['close'].shift(1)) / df['close'].shift(1)

    # Volatility
    features['range'] = (df['high'] - df['low']) / df['close']
    features['volatility'] = features['close_return'].rolling(10).std()

    # Moving average distances
    for period in [5, 10, 20]:
        sma = df['close'].rolling(period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalize to 0-1

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / df['close']

    # Volume features
    features['volume_change'] = df['volume'].pct_change()
    vol_sma = df['volume'].rolling(10).mean()
    features['volume_ratio'] = df['volume'] / (vol_sma + 1e-10)

    # Order flow approximation
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-10
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    features['buy_pressure'] = lower_wick / total_range
    features['sell_pressure'] = upper_wick / total_range
    features['pressure_imbalance'] = features['buy_pressure'] - features['sell_pressure']

    return features.fillna(0).replace([np.inf, -np.inf], 0)


def prepare_multitimeframe_data(minute_df):
    """
    Prepare multi-timeframe data

    For each timestamp t:
    - Input 1: Last 30 minutes of 1-min features
    - Input 2: Last 30 x 5-min candles (150 min history)
    - Input 3: Last 30 x 15-min candles (450 min history)
    - Target: Direction and % change at t+60
    """
    print("Aggregating to different timeframes...")

    # Keep original minute data
    df_1min = minute_df.copy()
    features_1min = calculate_features_for_timeframe(df_1min)

    # Aggregate to 5-min
    df_5min = aggregate_to_timeframe(minute_df, period=5)
    features_5min = calculate_features_for_timeframe(df_5min)

    # Aggregate to 15-min
    df_15min = aggregate_to_timeframe(minute_df, period=15)
    features_15min = calculate_features_for_timeframe(df_15min)

    print(f"1-min candles: {len(df_1min):,}")
    print(f"5-min candles: {len(df_5min):,}")
    print(f"15-min candles: {len(df_15min):,}")

    # Normalize each timeframe
    print("Normalizing features...")
    norm_1min = rolling_zscore(features_1min, window=60)
    norm_5min = rolling_zscore(features_5min, window=60)
    norm_15min = rolling_zscore(features_15min, window=60)

    norm_1min = norm_1min.fillna(0).replace([np.inf, -np.inf], 0)
    norm_5min = norm_5min.fillna(0).replace([np.inf, -np.inf], 0)
    norm_15min = norm_15min.fillna(0).replace([np.inf, -np.inf], 0)

    feature_names = list(norm_1min.columns)
    num_features = len(feature_names)

    mat_1min = norm_1min.values.astype(np.float32)
    mat_5min = norm_5min.values.astype(np.float32)
    mat_15min = norm_15min.values.astype(np.float32)

    close_1min = df_1min['close'].values

    X_1min = []
    X_5min = []
    X_15min = []
    y_direction = []
    y_change = []

    print("Creating aligned samples...")

    # We need enough history: 450 minutes (30 x 15-min) + 60 min prediction
    # Start from minute 500 to be safe
    warmup = max(SEQUENCE_15MIN * 15, SEQUENCE_1MIN) + 100

    for i in range(warmup, len(mat_1min) - PREDICTION_HORIZON):
        # 1-min features
        if i < SEQUENCE_1MIN:
            continue
        seq_1min = mat_1min[i - SEQUENCE_1MIN:i]

        # 5-min features (need to find corresponding index)
        min_5_idx = i // 5
        if min_5_idx < SEQUENCE_5MIN or min_5_idx >= len(mat_5min):
            continue
        seq_5min = mat_5min[min_5_idx - SEQUENCE_5MIN:min_5_idx]

        # 15-min features
        min_15_idx = i // 15
        if min_15_idx < SEQUENCE_15MIN or min_15_idx >= len(mat_15min):
            continue
        seq_15min = mat_15min[min_15_idx - SEQUENCE_15MIN:min_15_idx]

        # Validate shapes
        if seq_1min.shape != (SEQUENCE_1MIN, num_features):
            continue
        if seq_5min.shape != (SEQUENCE_5MIN, num_features):
            continue
        if seq_15min.shape != (SEQUENCE_15MIN, num_features):
            continue

        # Skip if NaN
        if np.isnan(seq_1min).any() or np.isnan(seq_5min).any() or np.isnan(seq_15min).any():
            continue

        # Target
        current_price = close_1min[i - 1]
        future_price = close_1min[i + PREDICTION_HORIZON - 1]

        pct_change = (future_price - current_price) / current_price * 100
        direction = 1 if pct_change > 0 else 0

        X_1min.append(seq_1min)
        X_5min.append(seq_5min)
        X_15min.append(seq_15min)
        y_direction.append(direction)
        y_change.append(np.clip(pct_change, -10, 10))

    X_1min = np.array(X_1min, dtype=np.float32)
    X_5min = np.array(X_5min, dtype=np.float32)
    X_15min = np.array(X_15min, dtype=np.float32)
    y_direction = np.array(y_direction, dtype=np.float32)
    y_change = np.array(y_change, dtype=np.float32)

    print(f"\nCreated {len(X_1min):,} aligned samples")
    print(f"X_1min shape: {X_1min.shape}")
    print(f"X_5min shape: {X_5min.shape}")
    print(f"X_15min shape: {X_15min.shape}")

    up_pct = np.mean(y_direction) * 100
    print(f"Class balance: UP={up_pct:.1f}%, DOWN={100-up_pct:.1f}%")

    return X_1min, X_5min, X_15min, y_direction, y_change, num_features


def build_cnn_head(input_shape, name):
    """Build a CNN head for one timeframe"""
    inputs = layers.Input(shape=input_shape, name=f'{name}_input')

    # 1D Convolutions to extract temporal patterns
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layer for this timeframe
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    return Model(inputs=inputs, outputs=x, name=f'{name}_cnn')


def build_multitimeframe_model(num_features):
    """
    Build multi-timeframe CNN with soft attention

    Architecture:
    - Three CNN heads (1-min, 5-min, 15-min)
    - Soft attention to weight timeframes
    - Final classification/regression heads
    """
    # Inputs for each timeframe
    input_1min = layers.Input(shape=(SEQUENCE_1MIN, num_features), name='input_1min')
    input_5min = layers.Input(shape=(SEQUENCE_5MIN, num_features), name='input_5min')
    input_15min = layers.Input(shape=(SEQUENCE_15MIN, num_features), name='input_15min')

    # CNN heads for each timeframe
    cnn_1min = build_cnn_head((SEQUENCE_1MIN, num_features), 'tf_1min')
    cnn_5min = build_cnn_head((SEQUENCE_5MIN, num_features), 'tf_5min')
    cnn_15min = build_cnn_head((SEQUENCE_15MIN, num_features), 'tf_15min')

    # Get embeddings from each head
    embed_1min = cnn_1min(input_1min)  # (batch, 64)
    embed_5min = cnn_5min(input_5min)  # (batch, 64)
    embed_15min = cnn_15min(input_15min)  # (batch, 64)

    # Stack embeddings for attention
    stacked = layers.Lambda(
        lambda x: tf.stack(x, axis=1)
    )([embed_1min, embed_5min, embed_15min])  # (batch, 3, 64)

    # Soft attention mechanism
    # Learn attention weights based on market conditions
    attention_dense = layers.Dense(32, activation='relu')(stacked)
    attention_weights = layers.Dense(1, activation='softmax')(attention_dense)  # (batch, 3, 1)
    attention_weights = layers.Flatten()(attention_weights)
    attention_weights = layers.Softmax()(attention_weights)  # (batch, 3)

    # Apply attention (weighted sum of embeddings)
    attention_weights_exp = layers.Lambda(
        lambda x: tf.expand_dims(x, -1)
    )(attention_weights)  # (batch, 3, 1)

    weighted = layers.Multiply()([stacked, attention_weights_exp])  # (batch, 3, 64)
    combined = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)  # (batch, 64)

    # Also concatenate for additional info
    concat = layers.Concatenate()([embed_1min, embed_5min, embed_15min])  # (batch, 192)

    # Combine attention output with concatenation
    merged = layers.Concatenate()([combined, concat])  # (batch, 256)

    # Shared dense layers
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output heads
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(x)
    price_output = layers.Dense(1, activation='linear', name='price_change')(x)
    confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)

    model = Model(
        inputs=[input_1min, input_5min, input_15min],
        outputs=[direction_output, price_output, confidence_output]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'direction': 'binary_crossentropy',
            'price_change': 'mse',
            'confidence': 'binary_crossentropy'
        },
        loss_weights={
            'direction': 1.0,
            'price_change': 0.1,
            'confidence': 0.5
        },
        metrics={
            'direction': 'accuracy',
            'price_change': 'mae'
        }
    )

    return model


def main():
    print("=" * 60)
    print("BTC MULTI-TIMEFRAME PREDICTOR")
    print("Based on: Multi-Head CNN with Soft Attention")
    print("=" * 60)
    print(f"Timeframes: 1-min ({SEQUENCE_1MIN}), 5-min ({SEQUENCE_5MIN}), 15-min ({SEQUENCE_15MIN})")
    print(f"Prediction horizon: {PREDICTION_HORIZON} minutes")
    print()

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    print()

    # Load data
    print("Loading BTC candles...")
    minute_df = load_all_btc_candles()
    print(f"Loaded {len(minute_df):,} minute candles")
    print()

    # Prepare multi-timeframe data
    X_1min, X_5min, X_15min, y_dir, y_chg, num_features = prepare_multitimeframe_data(minute_df)
    print()

    # Create confidence labels (1 if prediction was correct, useful for training)
    # This is a placeholder - in practice, we'd use actual confidence
    y_conf = np.abs(y_chg) > 0.5  # High confidence if move > 0.5%
    y_conf = y_conf.astype(np.float32)

    # Limit samples
    if len(X_1min) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        indices = np.random.choice(len(X_1min), MAX_SAMPLES, replace=False)
        X_1min = X_1min[indices]
        X_5min = X_5min[indices]
        X_15min = X_15min[indices]
        y_dir = y_dir[indices]
        y_chg = y_chg[indices]
        y_conf = y_conf[indices]

    # Split (temporal)
    split_idx = int(len(X_1min) * 0.8)

    X_1min_train, X_1min_test = X_1min[:split_idx], X_1min[split_idx:]
    X_5min_train, X_5min_test = X_5min[:split_idx], X_5min[split_idx:]
    X_15min_train, X_15min_test = X_15min[:split_idx], X_15min[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
    y_chg_train, y_chg_test = y_chg[:split_idx], y_chg[split_idx:]
    y_conf_train, y_conf_test = y_conf[:split_idx], y_conf[split_idx:]

    print(f"Train: {len(X_1min_train):,}, Test: {len(X_1min_test):,}")
    print()

    # Build model
    print("Building multi-timeframe model...")
    model = build_multitimeframe_model(num_features)
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
            model_path,
            monitor='val_direction_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = model.fit(
        [X_1min_train, X_5min_train, X_15min_train],
        {'direction': y_dir_train, 'price_change': y_chg_train, 'confidence': y_conf_train},
        validation_data=(
            [X_1min_test, X_5min_test, X_15min_test],
            {'direction': y_dir_test, 'price_change': y_chg_test, 'confidence': y_conf_test}
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
        [X_1min_test, X_5min_test, X_15min_test],
        {'direction': y_dir_test, 'price_change': y_chg_test, 'confidence': y_conf_test},
        verbose=0
    )

    # results order: loss, direction_loss, price_loss, conf_loss, dir_acc, price_mae
    print(f"Direction Accuracy: {results[4] * 100:.1f}%")
    print(f"Price MAE: {results[5]:.4f}%")

    # Detailed analysis
    predictions = model.predict(
        [X_1min_test, X_5min_test, X_15min_test],
        verbose=0
    )

    y_dir_pred = (predictions[0] > 0.5).astype(int).flatten()
    y_conf_pred = predictions[2].flatten()

    # High confidence predictions
    high_conf_mask = y_conf_pred > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_acc = np.mean(y_dir_pred[high_conf_mask] == y_dir_test[high_conf_mask])
        print(f"\nHigh confidence (>70%) accuracy: {high_conf_acc * 100:.1f}%")
        print(f"  ({high_conf_mask.sum()} predictions)")

    # Save config
    config = {
        'sequence_1min': SEQUENCE_1MIN,
        'sequence_5min': SEQUENCE_5MIN,
        'sequence_15min': SEQUENCE_15MIN,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': num_features,
        'direction_accuracy': float(results[4]),
        'price_mae': float(results[5]),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print(f"MULTI-TIMEFRAME DIRECTION ACCURACY: {results[4] * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
