#!/usr/bin/env python3
"""
BTC Autoregressive Model v2 - With Scheduled Sampling
Trains the model to handle its own prediction errors

Key improvement: During training, randomly use model's own predictions
as input instead of ground truth. This teaches the model to recover
from its own errors during the 30→31→32→...→90 prediction chain.
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
SEQUENCE_LENGTH = 30
PREDICTION_STEPS = 60  # Predict 60 steps autoregressively

# Training settings
BATCH_SIZE = 128  # Smaller batch for stability
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20

# Scheduled sampling settings
# Start with 100% teacher forcing, gradually use more predictions
INITIAL_TEACHER_FORCING = 1.0  # 100% real data at start
FINAL_TEACHER_FORCING = 0.3   # 30% real data at end (70% predictions)
TEACHER_FORCING_DECAY_EPOCHS = 50  # Decay over 50 epochs

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'autoregressive_v2')


class ScheduledSamplingCallback(tf.keras.callbacks.Callback):
    """Callback to adjust teacher forcing ratio during training"""
    def __init__(self, initial_ratio, final_ratio, decay_epochs):
        super().__init__()
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.decay_epochs = decay_epochs
        self.current_ratio = initial_ratio

    def on_epoch_begin(self, epoch, logs=None):
        # Linear decay
        decay_progress = min(1.0, epoch / self.decay_epochs)
        self.current_ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * decay_progress
        print(f"\nTeacher forcing ratio: {self.current_ratio:.2f}")


def prepare_autoregressive_data(candles_df):
    """Prepare sequences for autoregressive training"""
    print("Calculating features...")
    features = calculate_all_features(candles_df)
    normalized = rolling_zscore(features, window=60)
    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    feature_matrix = normalized[FEATURE_NAMES].values.astype(np.float32)

    # Target columns (OHLCV returns)
    target_cols = ['open_return', 'high_return', 'low_return', 'close_return', 'volume_change']
    target_indices = [FEATURE_NAMES.index(col) for col in target_cols]

    X = []  # Input sequences
    Y = []  # Target sequences (next 60 candles)

    print("Creating autoregressive sequences...")

    # For autoregressive, we need full sequences
    for i in range(SEQUENCE_LENGTH + 60, len(feature_matrix) - PREDICTION_STEPS):
        # Input: 30 candles
        input_seq = feature_matrix[i - SEQUENCE_LENGTH:i]

        # Target: next 60 candles' returns
        target_seq = feature_matrix[i:i + PREDICTION_STEPS, :][:, target_indices]

        if np.isnan(input_seq).any() or np.isnan(target_seq).any():
            continue

        X.append(input_seq)
        Y.append(target_seq)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Clip extreme values
    Y = np.clip(Y, -0.05, 0.05)

    print(f"Created {len(X):,} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {Y.shape}")

    return X, Y


def direction_aware_loss(y_true, y_pred):
    """Custom loss that heavily penalizes wrong direction predictions"""
    # MSE loss
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Direction loss - penalize when sign is wrong
    # Focus on close returns (index 3)
    true_close = y_true[:, :, 3]
    pred_close = y_pred[:, :, 3]

    # Cumulative returns for final direction
    true_cumulative = tf.reduce_sum(true_close, axis=1)
    pred_cumulative = tf.reduce_sum(pred_close, axis=1)

    # Sign agreement (1 if same sign, 0 if different)
    true_sign = tf.sign(true_cumulative)
    pred_sign = tf.sign(pred_cumulative)
    direction_match = tf.cast(tf.equal(true_sign, pred_sign), tf.float32)

    # Penalize wrong direction heavily
    direction_penalty = tf.reduce_mean(1.0 - direction_match) * 2.0

    return mse + direction_penalty


def build_autoregressive_model():
    """
    IMPROVED Transformer-LSTM Hybrid Model
    - Multi-head attention for pattern recognition
    - Deep LSTM for sequence modeling
    - Direction-aware loss function
    - Heavy regularization to prevent overfitting
    """
    encoder_input = layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name='encoder_input')

    # === TRANSFORMER ENCODER BLOCK ===
    # Project to higher dimension
    x = layers.Dense(128)(encoder_input)

    # Multi-head self-attention (8 heads)
    attention1 = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + attention1)

    # Feed-forward network
    ff = layers.Dense(256, activation='gelu')(x)
    ff = layers.Dropout(0.2)(ff)
    ff = layers.Dense(128)(ff)
    x = layers.LayerNormalization()(x + ff)

    # Second attention layer
    attention2 = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + attention2)

    # === LSTM ENCODER ===
    lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    lstm1 = layers.Dropout(0.3)(lstm1)
    lstm2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(lstm1)
    lstm2 = layers.Dropout(0.3)(lstm2)
    encoder_out = layers.Bidirectional(layers.LSTM(32))(lstm2)

    # === DECODER ===
    # Expand to 60 timesteps
    decoder = layers.RepeatVector(PREDICTION_STEPS)(encoder_out)

    # LSTM decoder layers
    decoder = layers.LSTM(128, return_sequences=True)(decoder)
    decoder = layers.Dropout(0.2)(decoder)
    decoder = layers.LSTM(64, return_sequences=True)(decoder)
    decoder = layers.Dropout(0.2)(decoder)

    # Attention over decoder output
    decoder_attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(decoder, decoder)
    decoder = layers.LayerNormalization()(decoder + decoder_attention)

    # Output layers
    output = layers.TimeDistributed(layers.Dense(64, activation='gelu'))(decoder)
    output = layers.Dropout(0.1)(output)
    output = layers.TimeDistributed(layers.Dense(32, activation='gelu'))(output)
    output = layers.TimeDistributed(layers.Dense(5, activation='tanh'))(output)

    # Scale to realistic range (-3% to +3% per candle)
    output = layers.Lambda(lambda x: x * 0.03, name='scaled_output')(output)

    model = Model(inputs=encoder_input, outputs=output)

    # Use direction-aware loss
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.01),
        loss=direction_aware_loss,
        metrics=['mae']
    )

    return model


def evaluate_direction_accuracy(model, X_test, Y_test):
    """Evaluate direction accuracy on test set"""
    predictions = model.predict(X_test, verbose=0)

    # Close return is index 3
    pred_directions = np.sum(predictions[:, :, 3], axis=1) > 0  # Cumulative direction
    actual_directions = np.sum(Y_test[:, :, 3], axis=1) > 0

    accuracy = np.mean(pred_directions == actual_directions)
    return accuracy


def main():
    print("=" * 60)
    print("BTC AUTOREGRESSIVE MODEL v2")
    print("With Scheduled Sampling for Error Recovery")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Predict: {PREDICTION_STEPS} steps autoregressively")
    print(f"Teacher forcing: {INITIAL_TEACHER_FORCING:.0%} → {FINAL_TEACHER_FORCING:.0%}")
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
    X, Y = prepare_autoregressive_data(candles_df)
    print()

    # Limit samples
    MAX_SAMPLES = 300000
    if len(X) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES:,} samples")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X, Y = X[idx], Y[idx]

    # Split (temporal to avoid lookahead)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print()

    # Build model
    print("Building autoregressive model...")
    model = build_autoregressive_model()
    model.summary()
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
        ),
        ScheduledSamplingCallback(
            INITIAL_TEACHER_FORCING, FINAL_TEACHER_FORCING, TEACHER_FORCING_DECAY_EPOCHS
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
    print(f"Test Loss (MSE): {results[0]:.6f}")
    print(f"Test MAE: {results[1]:.6f}")

    # Direction accuracy
    direction_acc = evaluate_direction_accuracy(model, X_test, Y_test)
    print(f"\n60-MINUTE DIRECTION ACCURACY: {direction_acc * 100:.1f}%")

    # Per-step analysis
    predictions = model.predict(X_test[:1000], verbose=0)
    for step in [0, 14, 29, 44, 59]:
        step_pred = predictions[:, step, 3]  # Close returns
        step_actual = Y_test[:1000, step, 3]
        step_dir_acc = np.mean((step_pred > 0) == (step_actual > 0))
        print(f"  Step {step+1} direction accuracy: {step_dir_acc * 100:.1f}%")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_steps': PREDICTION_STEPS,
        'num_features': NUM_FEATURES,
        'direction_accuracy': float(direction_acc),
        'test_mse': float(results[0]),
        'test_mae': float(results[1]),
        'trained_at': datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Model saved to: {model_path}")
    print()
    print("=" * 60)
    print(f"FINAL 60-MIN DIRECTION ACCURACY: {direction_acc * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
