#!/usr/bin/env python3
"""
BTC Price Predictor - Direct Price Change Prediction
Predicts both DIRECTION and EXACT PRICE CHANGE in 60 minutes

Key differences from autoregressive approach:
- Single prediction (no compounding errors)
- Predicts price change directly (not step-by-step)
- Multi-output: direction probability + expected % change
- Uses all 20 features + order flow (when available)
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

# Model settings
SEQUENCE_LENGTH = 30      # Use 30 minutes of input data
PREDICTION_HORIZON = 60   # Predict 60 minutes ahead

# Training settings
BATCH_SIZE = 256
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
MAX_SAMPLES = 400000  # Limit samples to avoid OOM

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'price_predictor')


def calculate_order_flow_features(df):
    """
    Calculate simulated order flow features from OHLCV data
    Real order flow requires tick data - this is an approximation
    """
    features = pd.DataFrame(index=df.index)

    # Buy/Sell pressure approximation using candle body position
    body = df['close'] - df['open']
    total_range = df['high'] - df['low']

    # Upper wick (selling pressure)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    # Lower wick (buying pressure)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    # Buy/sell pressure ratio
    features['buy_pressure'] = lower_wick / (total_range + 1e-10)
    features['sell_pressure'] = upper_wick / (total_range + 1e-10)
    features['pressure_imbalance'] = features['buy_pressure'] - features['sell_pressure']

    # Volume-weighted pressure
    features['volume_buy_pressure'] = features['buy_pressure'] * df['volume']
    features['volume_sell_pressure'] = features['sell_pressure'] * df['volume']

    # Cumulative delta approximation (bullish vs bearish candles)
    bullish = (df['close'] > df['open']).astype(float)
    bearish = (df['close'] < df['open']).astype(float)
    features['cumulative_delta'] = (bullish * df['volume'] - bearish * df['volume']).rolling(20).sum()
    features['cumulative_delta_norm'] = features['cumulative_delta'] / df['volume'].rolling(20).sum()

    # Large trade detection (volume spikes)
    vol_sma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    features['volume_zscore'] = (df['volume'] - vol_sma) / (vol_std + 1e-10)
    features['large_trade_count'] = (features['volume_zscore'] > 2).rolling(10).sum()

    # Momentum confirmation (price move with volume)
    price_change = df['close'].pct_change()
    features['momentum_volume'] = price_change * features['volume_zscore']

    return features


def prepare_prediction_data(candles_df, sequence_length=30, horizon=60):
    """
    Prepare data for direct price prediction

    For each timestamp t:
    - Input: Features from t-sequence_length to t
    - Target: (direction, percent_change) at t+horizon
    """
    print("Calculating standard features...")
    features = calculate_all_features(candles_df)

    print("Calculating order flow features...")
    order_flow = calculate_order_flow_features(candles_df)

    # Combine features
    all_features = pd.concat([features, order_flow], axis=1)

    print("Applying rolling z-score normalization...")
    normalized = rolling_zscore(all_features, window=60)
    normalized = normalized.fillna(0)

    # Replace infinities
    normalized = normalized.replace([np.inf, -np.inf], 0)

    feature_names = list(normalized.columns)
    feature_matrix = normalized.values

    print(f"Total features: {len(feature_names)}")

    # Get close prices for target calculation
    close_prices = candles_df['close'].values

    X = []
    y_direction = []
    y_change = []

    print(f"Creating sequences...")

    for i in range(sequence_length, len(feature_matrix) - horizon):
        # Input sequence
        seq = feature_matrix[i - sequence_length:i]

        # Skip if any NaN or Inf
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue

        # Current price and future price
        current_price = close_prices[i - 1]
        future_price = close_prices[i + horizon - 1]

        # Calculate percent change
        percent_change = (future_price - current_price) / current_price * 100  # In percent

        # Direction (1 = UP, 0 = DOWN)
        direction = 1 if percent_change > 0 else 0

        X.append(seq)
        y_direction.append(direction)
        y_change.append(percent_change)

    X = np.array(X, dtype=np.float32)
    y_direction = np.array(y_direction, dtype=np.float32)
    y_change = np.array(y_change, dtype=np.float32)

    # Clip extreme values
    y_change = np.clip(y_change, -10, 10)  # Clip to +/- 10%

    # Stats
    up_count = np.sum(y_direction == 1)
    down_count = np.sum(y_direction == 0)
    print(f"Samples: {len(X):,}")
    print(f"Class balance: UP={up_count} ({up_count/len(y_direction)*100:.1f}%), DOWN={down_count} ({down_count/len(y_direction)*100:.1f}%)")
    print(f"Price change stats: mean={np.mean(y_change):.3f}%, std={np.std(y_change):.3f}%")

    return X, y_direction, y_change, feature_names


def build_hybrid_model(num_features):
    """
    Build a hybrid model that predicts:
    1. Direction probability (classification)
    2. Expected price change (regression)
    """
    # Input
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, num_features))

    # Shared LSTM backbone
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)

    # Shared dense
    shared = layers.Dense(64, activation='relu')(x)
    shared = layers.Dropout(0.2)(shared)

    # Direction head (classification)
    direction_head = layers.Dense(32, activation='relu')(shared)
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(direction_head)

    # Price change head (regression)
    price_head = layers.Dense(32, activation='relu')(shared)
    price_output = layers.Dense(1, activation='linear', name='price_change')(price_head)

    # Build model
    model = Model(inputs=inputs, outputs=[direction_output, price_output])

    # Compile with multiple losses
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'direction': 'binary_crossentropy',
            'price_change': 'mse'
        },
        loss_weights={
            'direction': 1.0,  # Direction accuracy is important
            'price_change': 0.1  # Scale down MSE loss
        },
        metrics={
            'direction': 'accuracy',
            'price_change': 'mae'
        }
    )

    return model


def main():
    print("=" * 60)
    print("BTC PRICE PREDICTOR TRAINING")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Predict: Direction + Price change at {PREDICTION_HORIZON} minutes")
    print()

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"GPU memory config error: {e}")
    else:
        print("No GPU found, using CPU")
    print()

    # Load data
    print("Loading BTC candles...")
    candles_df = load_all_btc_candles()
    print(f"Loaded {len(candles_df):,} candles")
    print()

    # Prepare data
    X, y_direction, y_change, feature_names = prepare_prediction_data(
        candles_df, SEQUENCE_LENGTH, PREDICTION_HORIZON
    )
    print()

    # Limit samples to avoid OOM
    if len(X) > MAX_SAMPLES:
        print(f"Limiting samples from {len(X):,} to {MAX_SAMPLES:,}")
        indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[indices]
        y_direction = y_direction[indices]
        y_change = y_change[indices]
        print()

    # Split data
    X_train, X_test, y_dir_train, y_dir_test, y_chg_train, y_chg_test = train_test_split(
        X, y_direction, y_change, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print()

    # Build model
    print("Building hybrid model...")
    num_features = X.shape[2]
    model = build_hybrid_model(num_features)
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
            verbose=1,
            mode='max'  # Higher accuracy is better
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_direction_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        {'direction': y_dir_train, 'price_change': y_chg_train},
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

    # Final evaluation
    results = model.evaluate(
        X_test,
        {'direction': y_dir_test, 'price_change': y_chg_test},
        verbose=0
    )

    print(f"Test Loss: {results[0]:.4f}")
    print(f"Direction Accuracy: {results[3] * 100:.1f}%")
    print(f"Price Change MAE: {results[4]:.3f}%")

    # Detailed predictions
    predictions = model.predict(X_test, verbose=0)
    y_dir_pred = (predictions[0] > 0.5).astype(int).flatten()
    y_chg_pred = predictions[1].flatten()

    # Direction accuracy breakdown
    print()
    print("Direction Accuracy Breakdown:")

    # When predicting UP
    pred_up_mask = y_dir_pred == 1
    if pred_up_mask.sum() > 0:
        up_correct = (y_dir_test[pred_up_mask] == 1).sum()
        up_acc = up_correct / pred_up_mask.sum() * 100
        print(f"  When predicting UP: {up_acc:.1f}% correct ({pred_up_mask.sum()} predictions)")

    # When predicting DOWN
    pred_down_mask = y_dir_pred == 0
    if pred_down_mask.sum() > 0:
        down_correct = (y_dir_test[pred_down_mask] == 0).sum()
        down_acc = down_correct / pred_down_mask.sum() * 100
        print(f"  When predicting DOWN: {down_acc:.1f}% correct ({pred_down_mask.sum()} predictions)")

    # Price prediction accuracy
    print()
    print("Price Change Prediction:")
    print(f"  Predicted range: {y_chg_pred.min():.2f}% to {y_chg_pred.max():.2f}%")
    print(f"  Actual range: {y_chg_test.min():.2f}% to {y_chg_test.max():.2f}%")

    # Correlation between predicted and actual
    correlation = np.corrcoef(y_chg_pred, y_chg_test)[0, 1]
    print(f"  Prediction-Actual Correlation: {correlation:.3f}")

    # Price accuracy when direction is correct
    correct_mask = y_dir_pred == y_dir_test
    if correct_mask.sum() > 0:
        mae_correct = np.mean(np.abs(y_chg_pred[correct_mask] - y_chg_test[correct_mask]))
        print(f"  MAE when direction correct: {mae_correct:.3f}%")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': num_features,
        'feature_names': feature_names,
        'direction_accuracy': float(results[3]),
        'price_mae': float(results[4]),
        'price_correlation': float(correlation),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'trained_at': datetime.now().isoformat()
    }

    config_path = os.path.join(MODEL_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print(f"Config saved to: {config_path}")
    print()
    print("=" * 60)
    print(f"DIRECTION ACCURACY: {results[3] * 100:.1f}%")
    print(f"PRICE CHANGE MAE: {results[4]:.3f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
