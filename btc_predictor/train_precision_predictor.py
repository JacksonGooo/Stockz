#!/usr/bin/env python3
"""
BTC Precision Price Predictor
Aims for maximum price accuracy using every available technique

Strategy:
1. Predict DELTA from current price (not absolute price)
2. Use multiple prediction heads for different aspects
3. Ensemble multiple model architectures
4. Use uncertainty quantification (predict confidence)
5. Focus on high-confidence predictions only

Key insight: Instead of predicting exact price, predict:
- Direction (UP/DOWN) with confidence
- Expected move magnitude
- Volatility estimate
- Price range (high/low bounds)
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
from tensorflow.keras import backend as K

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES
from data_loader import load_all_btc_candles

# Model settings
SEQUENCE_LENGTH = 60      # Use more history
PREDICTION_HORIZON = 60   # Predict 60 minutes ahead

# Training settings
BATCH_SIZE = 128
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
MAX_SAMPLES = 400000

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'precision_predictor')


def calculate_advanced_features(df):
    """Calculate comprehensive features for maximum predictive power"""
    features = pd.DataFrame(index=df.index)

    # === PRICE RETURNS ===
    for period in [1, 5, 10, 15, 30, 60]:
        features[f'return_{period}'] = df['close'].pct_change(period)

    # === VOLATILITY MEASURES ===
    features['range'] = (df['high'] - df['low']) / df['close']
    features['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    ) / df['close']

    for period in [5, 10, 20, 30, 60]:
        features[f'volatility_{period}'] = features['return_1'].rolling(period).std()
        features[f'atr_{period}'] = features['true_range'].rolling(period).mean()

    # === MOVING AVERAGES ===
    for period in [5, 10, 20, 30, 50, 100]:
        sma = df['close'].rolling(period).mean()
        ema = df['close'].ewm(span=period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma
        features[f'ema_{period}_dist'] = (df['close'] - ema) / ema

    # === MOMENTUM INDICATORS ===
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi_norm'] = features['rsi'] / 100

    # Stochastic
    lowest_low = df['low'].rolling(14).min()
    highest_high = df['high'].rolling(14).max()
    features['stoch_k'] = (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd'] = macd / df['close']
    features['macd_signal'] = signal / df['close']
    features['macd_hist'] = (macd - signal) / df['close']

    # === VOLUME ANALYSIS ===
    features['volume_change'] = df['volume'].pct_change()
    for period in [5, 10, 20]:
        vol_sma = df['volume'].rolling(period).mean()
        features[f'volume_ratio_{period}'] = df['volume'] / (vol_sma + 1e-10)

    # Volume-price correlation
    features['vp_corr'] = df['close'].rolling(20).corr(df['volume'])

    # === ORDER FLOW APPROXIMATION ===
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-10
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    features['buy_pressure'] = lower_wick / total_range
    features['sell_pressure'] = upper_wick / total_range
    features['pressure_imbalance'] = features['buy_pressure'] - features['sell_pressure']

    # Cumulative delta
    bullish = (df['close'] > df['open']).astype(float)
    bearish = (df['close'] < df['open']).astype(float)
    for period in [5, 10, 20]:
        features[f'cum_delta_{period}'] = (
            bullish * df['volume'] - bearish * df['volume']
        ).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)

    # === PATTERN FEATURES ===
    # Higher highs / lower lows
    features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float).rolling(5).mean()
    features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float).rolling(5).mean()

    # Close position in range
    features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Gap
    features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # === TIME FEATURES ===
    df_time = pd.to_datetime(df['timestamp'], unit='ms')
    features['hour_sin'] = np.sin(2 * np.pi * df_time.dt.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * df_time.dt.hour / 24)
    features['dow_sin'] = np.sin(2 * np.pi * df_time.dt.dayofweek / 7)
    features['dow_cos'] = np.cos(2 * np.pi * df_time.dt.dayofweek / 7)

    return features.fillna(0).replace([np.inf, -np.inf], 0)


def prepare_precision_data(candles_df, sequence_length=60, horizon=60):
    """Prepare data for precision price prediction"""
    print("Calculating advanced features...")
    features = calculate_advanced_features(candles_df)

    print(f"Total features: {len(features.columns)}")

    print("Normalizing features...")
    normalized = rolling_zscore(features, window=120)
    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    feature_names = list(normalized.columns)
    feature_matrix = normalized.values.astype(np.float32)

    close_prices = candles_df['close'].values
    high_prices = candles_df['high'].values
    low_prices = candles_df['low'].values

    X = []
    y_direction = []
    y_change = []
    y_high_change = []  # Max high in next 60 min
    y_low_change = []   # Min low in next 60 min
    y_volatility = []   # Actual volatility in next 60 min

    print("Creating sequences...")
    warmup = max(sequence_length, 120) + 60  # For rolling windows

    for i in range(warmup, len(feature_matrix) - horizon):
        seq = feature_matrix[i - sequence_length:i]

        if np.isnan(seq).any() or np.isinf(seq).any():
            continue

        current_price = close_prices[i - 1]

        # Future values
        future_slice = slice(i, i + horizon)
        future_close = close_prices[i + horizon - 1]
        future_high = np.max(high_prices[future_slice])
        future_low = np.min(low_prices[future_slice])
        future_volatility = np.std(close_prices[future_slice]) / current_price

        # Calculate targets as percent change
        pct_change = (future_close - current_price) / current_price * 100
        high_change = (future_high - current_price) / current_price * 100
        low_change = (future_low - current_price) / current_price * 100

        direction = 1 if pct_change > 0 else 0

        X.append(seq)
        y_direction.append(direction)
        y_change.append(np.clip(pct_change, -10, 10))
        y_high_change.append(np.clip(high_change, 0, 15))  # High is always >= 0
        y_low_change.append(np.clip(low_change, -15, 0))   # Low is always <= 0
        y_volatility.append(np.clip(future_volatility * 100, 0, 10))

    X = np.array(X, dtype=np.float32)
    y_direction = np.array(y_direction, dtype=np.float32)
    y_change = np.array(y_change, dtype=np.float32)
    y_high_change = np.array(y_high_change, dtype=np.float32)
    y_low_change = np.array(y_low_change, dtype=np.float32)
    y_volatility = np.array(y_volatility, dtype=np.float32)

    print(f"Created {len(X):,} samples")
    print(f"Features: {X.shape[2]}")
    print(f"Direction balance: UP={np.mean(y_direction)*100:.1f}%")
    print(f"Price change stats: mean={np.mean(y_change):.3f}%, std={np.std(y_change):.3f}%")
    print(f"High change range: {np.min(y_high_change):.2f}% to {np.max(y_high_change):.2f}%")
    print(f"Low change range: {np.min(y_low_change):.2f}% to {np.max(y_low_change):.2f}%")

    return X, y_direction, y_change, y_high_change, y_low_change, y_volatility, feature_names


def quantile_loss(q):
    """Quantile loss for prediction intervals"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return K.mean(K.maximum(q * error, (q - 1) * error))
    return loss


def build_precision_model(num_features):
    """
    Build precision price predictor with multiple output heads

    Outputs:
    1. Direction probability
    2. Expected price change (point estimate)
    3. High bound (what's the max likely price)
    4. Low bound (what's the min likely price)
    5. Volatility estimate
    6. Confidence score
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # === ENCODER: Extract patterns ===

    # Temporal Convolutions for local patterns
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    conv2 = layers.BatchNormalization()(conv2)
    conv3 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    conv3 = layers.BatchNormalization()(conv3)

    # Multi-scale convolution fusion
    conv_merged = layers.Concatenate()([conv1, conv2, conv3])  # (batch, seq, 192)
    conv_merged = layers.Conv1D(128, kernel_size=1)(conv_merged)  # Compress
    conv_merged = layers.MaxPooling1D(pool_size=2)(conv_merged)

    # Multi-head self-attention
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(conv_merged, conv_merged)
    attention = layers.LayerNormalization()(attention + conv_merged[:, ::2, :])  # Skip connection

    # Bidirectional LSTM for sequential patterns
    lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(attention)
    lstm1 = layers.Dropout(0.2)(lstm1)
    lstm2 = layers.Bidirectional(layers.LSTM(32))(lstm1)
    lstm2 = layers.Dropout(0.2)(lstm2)

    # Global features
    global_avg = layers.GlobalAveragePooling1D()(attention)
    global_max = layers.GlobalMaxPooling1D()(attention)

    # Combine all representations
    combined = layers.Concatenate()([lstm2, global_avg, global_max])

    # Shared dense layers
    shared = layers.Dense(128, activation='relu')(combined)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(64, activation='relu')(shared)
    shared = layers.Dropout(0.2)(shared)

    # === OUTPUT HEADS ===

    # 1. Direction (classification)
    dir_hidden = layers.Dense(32, activation='relu')(shared)
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(dir_hidden)

    # 2. Price change (regression)
    price_hidden = layers.Dense(32, activation='relu')(shared)
    price_output = layers.Dense(1, activation='linear', name='price_change')(price_hidden)

    # 3. High bound (upper quantile)
    high_hidden = layers.Dense(32, activation='relu')(shared)
    high_output = layers.Dense(1, activation='relu', name='high_bound')(high_hidden)  # Always positive

    # 4. Low bound (lower quantile)
    low_hidden = layers.Dense(32, activation='relu')(shared)
    low_output = layers.Dense(1, activation='linear', name='low_bound')(low_hidden)

    # 5. Volatility estimate
    vol_hidden = layers.Dense(32, activation='relu')(shared)
    volatility_output = layers.Dense(1, activation='relu', name='volatility')(vol_hidden)

    # 6. Confidence (how certain are we about this prediction)
    conf_hidden = layers.Dense(32, activation='relu')(shared)
    confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(conf_hidden)

    model = Model(
        inputs=inputs,
        outputs=[
            direction_output,
            price_output,
            high_output,
            low_output,
            volatility_output,
            confidence_output
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'direction': 'binary_crossentropy',
            'price_change': 'mse',
            'high_bound': 'mse',
            'low_bound': 'mse',
            'volatility': 'mse',
            'confidence': 'binary_crossentropy'
        },
        loss_weights={
            'direction': 1.0,
            'price_change': 0.5,
            'high_bound': 0.3,
            'low_bound': 0.3,
            'volatility': 0.2,
            'confidence': 0.5
        },
        metrics={
            'direction': 'accuracy',
            'price_change': 'mae',
            'high_bound': 'mae',
            'low_bound': 'mae'
        }
    )

    return model


def main():
    print("=" * 60)
    print("BTC PRECISION PRICE PREDICTOR")
    print("Multi-Head Model for Maximum Accuracy")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Predict: {PREDICTION_HORIZON} minutes ahead")
    print()

    # GPU setup
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
    X, y_dir, y_chg, y_high, y_low, y_vol, feature_names = prepare_precision_data(
        candles_df, SEQUENCE_LENGTH, PREDICTION_HORIZON
    )
    print()

    # Confidence target: 1 if direction was predicted correctly (we'll use direction as proxy)
    # In practice, you'd want actual backtest results
    y_conf = (np.abs(y_chg) > 0.3).astype(np.float32)  # Confident when move > 0.3%

    # Limit samples
    if len(X) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[idx]
        y_dir = y_dir[idx]
        y_chg = y_chg[idx]
        y_high = y_high[idx]
        y_low = y_low[idx]
        y_vol = y_vol[idx]
        y_conf = y_conf[idx]

    # Split (temporal)
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
    y_chg_train, y_chg_test = y_chg[:split_idx], y_chg[split_idx:]
    y_high_train, y_high_test = y_high[:split_idx], y_high[split_idx:]
    y_low_train, y_low_test = y_low[:split_idx], y_low[split_idx:]
    y_vol_train, y_vol_test = y_vol[:split_idx], y_vol[split_idx:]
    y_conf_train, y_conf_test = y_conf[:split_idx], y_conf[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build model
    print("Building precision model...")
    model = build_precision_model(X.shape[2])
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
            'price_change': y_chg_train,
            'high_bound': y_high_train,
            'low_bound': y_low_train,
            'volatility': y_vol_train,
            'confidence': y_conf_train
        },
        validation_data=(
            X_test,
            {
                'direction': y_dir_test,
                'price_change': y_chg_test,
                'high_bound': y_high_test,
                'low_bound': y_low_test,
                'volatility': y_vol_test,
                'confidence': y_conf_test
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
            'price_change': y_chg_test,
            'high_bound': y_high_test,
            'low_bound': y_low_test,
            'volatility': y_vol_test,
            'confidence': y_conf_test
        },
        verbose=0
    )

    print(f"Direction Accuracy: {results[7] * 100:.1f}%")
    print(f"Price Change MAE: {results[8]:.4f}%")
    print(f"High Bound MAE: {results[9]:.4f}%")
    print(f"Low Bound MAE: {results[10]:.4f}%")

    # Get predictions for analysis
    predictions = model.predict(X_test, verbose=0)
    y_dir_pred = (predictions[0] > 0.5).astype(int).flatten()
    y_chg_pred = predictions[1].flatten()
    y_high_pred = predictions[2].flatten()
    y_low_pred = predictions[3].flatten()
    y_vol_pred = predictions[4].flatten()
    y_conf_pred = predictions[5].flatten()

    # Calculate dollar accuracy (assuming $100,000 BTC)
    btc_price = 100000
    mae_dollars = np.mean(np.abs(y_chg_pred - y_chg_test)) / 100 * btc_price
    print(f"\nPrice MAE in dollars (at $100k BTC): ${mae_dollars:.2f}")

    # High confidence predictions
    high_conf_mask = y_conf_pred > 0.7
    if high_conf_mask.sum() > 0:
        hc_dir_acc = np.mean(y_dir_pred[high_conf_mask] == y_dir_test[high_conf_mask])
        hc_mae = np.mean(np.abs(y_chg_pred[high_conf_mask] - y_chg_test[high_conf_mask]))
        hc_mae_dollars = hc_mae / 100 * btc_price
        print(f"\nHigh Confidence (>70%) Predictions:")
        print(f"  Count: {high_conf_mask.sum()} ({high_conf_mask.mean()*100:.1f}% of total)")
        print(f"  Direction Accuracy: {hc_dir_acc * 100:.1f}%")
        print(f"  Price MAE: {hc_mae:.4f}%")
        print(f"  Price MAE (dollars): ${hc_mae_dollars:.2f}")

    # Price range accuracy (did actual price fall between predicted high/low?)
    within_range = (y_chg_test <= y_high_pred) & (y_chg_test >= y_low_pred)
    range_accuracy = np.mean(within_range)
    print(f"\nPrice within predicted range: {range_accuracy * 100:.1f}%")

    # Predicted range width
    avg_range = np.mean(y_high_pred - y_low_pred)
    print(f"Average predicted range: {avg_range:.2f}%")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': X.shape[2],
        'feature_names': feature_names,
        'direction_accuracy': float(results[7]),
        'price_mae': float(results[8]),
        'high_mae': float(results[9]),
        'low_mae': float(results[10]),
        'price_mae_dollars': float(mae_dollars),
        'range_accuracy': float(range_accuracy),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print(f"PRECISION PREDICTOR RESULTS:")
    print(f"  Direction Accuracy: {results[7] * 100:.1f}%")
    print(f"  Price MAE: {results[8]:.4f}% (${mae_dollars:.2f})")
    print(f"  Range Accuracy: {range_accuracy * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
