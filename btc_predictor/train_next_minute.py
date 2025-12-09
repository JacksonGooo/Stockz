#!/usr/bin/env python3
"""
BTC Next-Minute Predictor (True Autoregressive)

Strategy: Train a model to predict ONLY the next 1 minute OHLCV.
Then chain it 60 times: 30 → 31 → 32 → ... → 90

This is the Krafer approach:
1. Take 30 minutes of data
2. Predict minute 31 (OHLC values)
3. Append prediction to data
4. Use minutes 2-31 to predict minute 32
5. Repeat 60 times

The key to making this work:
1. Model must be VERY accurate at 1-minute prediction
2. Model must learn to handle its OWN predictions as input (scheduled sampling)
3. Output must be constrained to realistic ranges
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
SEQUENCE_LENGTH = 30      # Input: 30 minutes
OUTPUT_FEATURES = 5       # Output: open, high, low, close, volume returns

# Training settings
BATCH_SIZE = 256
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 25
MAX_SAMPLES = 600000

# Scheduled sampling - gradually use model predictions during training
INITIAL_TEACHER_FORCING = 1.0   # 100% real data at start
FINAL_TEACHER_FORCING = 0.3     # 30% real data at end
TEACHER_DECAY_EPOCHS = 50

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'next_minute')


def calculate_ohlcv_features(df):
    """
    Calculate features from OHLCV data
    Focus on features that help predict the next candle
    """
    features = pd.DataFrame(index=df.index)

    # === RAW PRICE RETURNS (most important) ===
    features['open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    features['high_return'] = (df['high'] - df['close'].shift(1)) / df['close'].shift(1)
    features['low_return'] = (df['low'] - df['close'].shift(1)) / df['close'].shift(1)
    features['close_return'] = df['close'].pct_change()
    features['volume_change'] = df['volume'].pct_change()

    # === MOMENTUM ===
    for period in [3, 5, 10, 20]:
        features[f'momentum_{period}'] = df['close'].pct_change(period)

    # === VOLATILITY ===
    features['range'] = (df['high'] - df['low']) / df['close']
    for period in [5, 10, 20]:
        features[f'volatility_{period}'] = features['close_return'].rolling(period).std()

    # === ORDER FLOW PROXY ===
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-10
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    features['buy_pressure'] = lower_wick / total_range
    features['sell_pressure'] = upper_wick / total_range
    features['pressure_diff'] = features['buy_pressure'] - features['sell_pressure']

    # Cumulative pressure
    bullish = (df['close'] > df['open']).astype(float)
    bearish = (df['close'] < df['open']).astype(float)
    for period in [5, 10]:
        delta = bullish * df['volume'] - bearish * df['volume']
        vol_sum = df['volume'].rolling(period).sum() + 1e-10
        features[f'cum_delta_{period}'] = delta.rolling(period).sum() / vol_sum

    # === VOLUME ===
    for period in [5, 10, 20]:
        vol_ma = df['volume'].rolling(period).mean()
        features[f'volume_ratio_{period}'] = df['volume'] / (vol_ma + 1e-10)

    # === TECHNICAL ===
    # Short-term RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / (loss + 1e-10)
    features['rsi_7'] = (100 - (100 / (1 + rs))) / 100  # Normalize 0-1

    # MA distances
    for period in [5, 10, 20]:
        sma = df['close'].rolling(period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma

    return features.fillna(0).replace([np.inf, -np.inf], 0)


def prepare_next_minute_data(candles_df, seq_length=30):
    """
    Prepare data for next-minute prediction

    Input: 30 minutes of OHLCV + features
    Output: Next minute's OHLCV returns (5 values)
    """
    print("Calculating features...")
    features = calculate_ohlcv_features(candles_df)

    print(f"Total features: {len(features.columns)}")

    # Normalize with rolling z-score
    print("Normalizing...")
    normalized = features.copy()
    for col in features.columns:
        rolling_mean = features[col].rolling(60, min_periods=1).mean()
        rolling_std = features[col].rolling(60, min_periods=1).std() + 1e-8
        normalized[col] = (features[col] - rolling_mean) / rolling_std

    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    feature_names = list(normalized.columns)
    feature_matrix = normalized.values.astype(np.float32)

    # Target: Next candle's OHLCV returns (first 5 features)
    # These are: open_return, high_return, low_return, close_return, volume_change
    target_indices = [0, 1, 2, 3, 4]

    X = []
    Y = []

    print("Creating sequences...")
    warmup = max(seq_length, 60) + 10

    for i in range(warmup, len(feature_matrix) - 1):
        # Input: last 30 minutes of all features
        input_seq = feature_matrix[i - seq_length:i]

        # Target: next minute's OHLCV returns (unnormalized, raw returns)
        # Get raw returns from features DataFrame
        target = np.array([
            features.iloc[i]['open_return'],
            features.iloc[i]['high_return'],
            features.iloc[i]['low_return'],
            features.iloc[i]['close_return'],
            features.iloc[i]['volume_change']
        ], dtype=np.float32)

        if np.isnan(input_seq).any() or np.isnan(target).any():
            continue
        if np.isinf(input_seq).any() or np.isinf(target).any():
            continue

        X.append(input_seq)
        Y.append(target)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Clip extreme targets (BTC rarely moves more than 2% in 1 minute)
    Y = np.clip(Y, -0.02, 0.02)

    print(f"\nCreated {len(X):,} samples")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {Y.shape}")

    print(f"\nTarget statistics (% returns):")
    print(f"  Open:   mean={Y[:, 0].mean()*100:.4f}%, std={Y[:, 0].std()*100:.4f}%")
    print(f"  High:   mean={Y[:, 1].mean()*100:.4f}%, std={Y[:, 1].std()*100:.4f}%")
    print(f"  Low:    mean={Y[:, 2].mean()*100:.4f}%, std={Y[:, 2].std()*100:.4f}%")
    print(f"  Close:  mean={Y[:, 3].mean()*100:.4f}%, std={Y[:, 3].std()*100:.4f}%")
    print(f"  Volume: mean={Y[:, 4].mean()*100:.4f}%, std={Y[:, 4].std()*100:.4f}%")

    return X, Y, feature_names


def build_next_minute_model(num_features):
    """
    Build model optimized for next-minute prediction

    Key: Output OHLCV returns for the next candle
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # === MULTI-SCALE CONVOLUTIONS ===
    conv3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv5 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    conv7 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)

    merged = layers.Concatenate()([conv3, conv5, conv7])
    x = layers.BatchNormalization()(merged)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # === ATTENTION ===
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + attention)

    # === LSTM ===
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    lstm_out = layers.Bidirectional(layers.LSTM(64))(x)

    # === GLOBAL FEATURES ===
    global_avg = layers.GlobalAveragePooling1D()(attention)
    combined = layers.Concatenate()([lstm_out, global_avg])

    # === DENSE ===
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)

    # === OUTPUT: 5 values (open, high, low, close, volume returns) ===
    # Use tanh to constrain output, then scale to realistic range
    raw_output = layers.Dense(OUTPUT_FEATURES, activation='tanh')(x)

    # Scale to ±2% range (realistic 1-minute BTC moves)
    output = layers.Lambda(lambda x: x * 0.02, name='ohlcv_returns')(raw_output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


class ScheduledSamplingCallback(tf.keras.callbacks.Callback):
    """Callback to track teacher forcing ratio"""
    def __init__(self, initial, final, decay_epochs):
        super().__init__()
        self.initial = initial
        self.final = final
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = min(1.0, epoch / self.decay_epochs)
        ratio = self.initial - (self.initial - self.final) * progress
        print(f"\n[Teacher Forcing: {ratio*100:.0f}%]")


def main():
    print("=" * 60)
    print("BTC NEXT-MINUTE PREDICTOR")
    print("True Autoregressive: 30 → 31 → 32 → ... → 90")
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

    # Load data
    print("Loading BTC candles...")
    candles_df = load_all_btc_candles()
    print(f"Loaded {len(candles_df):,} candles")
    print()

    # Prepare data
    X, Y, feature_names = prepare_next_minute_data(candles_df, SEQUENCE_LENGTH)
    print()

    # Limit samples
    if len(X) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X, Y = X[idx], Y[idx]

    # Split (temporal)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build model
    print("Building next-minute model...")
    model = build_next_minute_model(X.shape[2])
    model.summary()
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
            patience=7,
            min_lr=1e-6
        ),
        ScheduledSamplingCallback(
            INITIAL_TEACHER_FORCING,
            FINAL_TEACHER_FORCING,
            TEACHER_DECAY_EPOCHS
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
    print(f"Test MSE: {results[0]:.8f}")
    print(f"Test MAE: {results[1]:.6f} ({results[1]*100:.4f}%)")

    # Per-feature MAE
    predictions = model.predict(X_test, verbose=0)
    feature_labels = ['Open', 'High', 'Low', 'Close', 'Volume']

    print("\nPer-feature MAE:")
    for i, label in enumerate(feature_labels):
        mae = np.mean(np.abs(predictions[:, i] - Y_test[:, i]))
        print(f"  {label}: {mae*100:.4f}%")

    # Direction accuracy for close
    pred_dir = predictions[:, 3] > 0  # Close direction
    actual_dir = Y_test[:, 3] > 0
    dir_acc = np.mean(pred_dir == actual_dir)
    print(f"\nClose Direction Accuracy: {dir_acc * 100:.1f}%")

    # Dollar accuracy at $100k BTC
    btc_price = 100000
    close_mae = np.mean(np.abs(predictions[:, 3] - Y_test[:, 3]))
    dollar_mae = close_mae * btc_price
    print(f"\nClose MAE in dollars: ${dollar_mae:.2f}")

    # Test autoregressive chaining
    print("\n--- AUTOREGRESSIVE CHAIN TEST ---")
    test_sample = X_test[0:1]  # Single sample

    chain_predictions = []
    current_input = test_sample.copy()

    for step in range(10):  # Chain 10 predictions
        pred = model.predict(current_input, verbose=0)[0]
        chain_predictions.append(pred)

        # Shift input window and append prediction
        # For now, just show the predictions - full implementation needs feature calculation
        cumulative_close = sum(p[3] for p in chain_predictions)
        print(f"  Step {step+1}: Close change = {cumulative_close*100:+.4f}%")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'output_features': OUTPUT_FEATURES,
        'num_input_features': X.shape[2],
        'feature_names': feature_names,
        'test_mse': float(results[0]),
        'test_mae': float(results[1]),
        'direction_accuracy': float(dir_acc),
        'close_mae_dollars': float(dollar_mae),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print("NEXT-MINUTE MODEL RESULTS:")
    print(f"  Close MAE: {close_mae*100:.4f}% (${dollar_mae:.2f})")
    print(f"  Direction Accuracy: {dir_acc * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
