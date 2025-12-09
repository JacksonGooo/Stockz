#!/usr/bin/env python3
"""
BTC Direction Classifier
Predicts whether BTC will go UP or DOWN in the next 60 minutes
Uses same 20 features as autoregressive model but outputs binary classification
No chaining = no compounding errors
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

from features import calculate_all_features, rolling_zscore, FEATURE_NAMES, NUM_FEATURES
from data_loader import load_all_btc_candles

# Model settings
SEQUENCE_LENGTH = 30      # Use 30 minutes of input data
PREDICTION_HORIZON = 60   # Predict 60 minutes ahead
MIN_MOVE_THRESHOLD = 0.001  # 0.1% minimum move to count as UP/DOWN (else neutral)

# Training settings
BATCH_SIZE = 512
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Output directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'direction_classifier')


def prepare_classification_data(candles_df, sequence_length=30, horizon=60):
    """
    Prepare data for direction classification

    For each timestamp t:
    - Input: Features from t-sequence_length to t
    - Target: 1 if price goes UP by horizon minutes, 0 if DOWN
    """
    print("Calculating features...")
    features = calculate_all_features(candles_df)
    normalized = rolling_zscore(features, window=60)
    normalized = normalized.fillna(0)
    feature_matrix = normalized[FEATURE_NAMES].values

    # Get close prices for determining direction
    close_prices = candles_df['close'].values

    X = []
    y = []

    # We need sequence_length + horizon candles for each sample
    total_needed = sequence_length + horizon

    print(f"Creating sequences (need {total_needed} candles per sample)...")

    for i in range(sequence_length, len(feature_matrix) - horizon):
        # Input sequence
        seq = feature_matrix[i - sequence_length:i]

        # Current price and future price
        current_price = close_prices[i - 1]  # Price at end of input sequence
        future_price = close_prices[i + horizon - 1]  # Price 60 minutes later

        # Calculate percent change
        percent_change = (future_price - current_price) / current_price

        # Determine direction (skip if move is too small)
        if abs(percent_change) < MIN_MOVE_THRESHOLD:
            continue  # Skip neutral samples

        direction = 1 if percent_change > 0 else 0  # 1 = UP, 0 = DOWN

        X.append(seq)
        y.append(direction)

    X = np.array(X)
    y = np.array(y)

    # Check class balance
    up_count = np.sum(y == 1)
    down_count = np.sum(y == 0)
    print(f"Class balance: UP={up_count} ({up_count/len(y)*100:.1f}%), DOWN={down_count} ({down_count/len(y)*100:.1f}%)")

    return X, y


def build_classifier_model():
    """
    Build LSTM classifier for direction prediction
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        # Bidirectional LSTM for better pattern recognition
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)
        ),
        tf.keras.layers.Dropout(0.3),

        # Second LSTM layer
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64)
        ),
        tf.keras.layers.Dropout(0.3),

        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),

        # Output: single neuron with sigmoid for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    print("=" * 60)
    print("BTC DIRECTION CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Input: {SEQUENCE_LENGTH} minutes")
    print(f"Predict: Direction after {PREDICTION_HORIZON} minutes")
    print(f"Min move threshold: {MIN_MOVE_THRESHOLD * 100}%")
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

    # Prepare classification data
    X, y = prepare_classification_data(candles_df, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    print(f"Total samples: {len(X):,}")
    print()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print()

    # Build model
    print("Building classifier model...")
    model = build_classifier_model()
    model.summary()
    print()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
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
        X_train, y_train,
        validation_data=(X_test, y_test),
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
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.1f}%")

    # Detailed predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Confusion matrix style analysis
    true_up_pred_up = np.sum((y_test == 1) & (y_pred == 1))
    true_up_pred_down = np.sum((y_test == 1) & (y_pred == 0))
    true_down_pred_up = np.sum((y_test == 0) & (y_pred == 1))
    true_down_pred_down = np.sum((y_test == 0) & (y_pred == 0))

    print()
    print("Confusion Matrix:")
    print(f"  Predicted UP when actually UP:   {true_up_pred_up} ({true_up_pred_up/(true_up_pred_up+true_up_pred_down)*100:.1f}%)")
    print(f"  Predicted DOWN when actually UP: {true_up_pred_down}")
    print(f"  Predicted UP when actually DOWN: {true_down_pred_up}")
    print(f"  Predicted DOWN when actually DOWN: {true_down_pred_down} ({true_down_pred_down/(true_down_pred_up+true_down_pred_down)*100:.1f}%)")

    # Accuracy when confident
    high_confidence = np.abs(y_pred_probs.flatten() - 0.5) > 0.2  # >70% or <30% confidence
    if np.sum(high_confidence) > 0:
        hc_accuracy = np.mean(y_pred[high_confidence] == y_test[high_confidence])
        print(f"\nHigh confidence predictions: {np.sum(high_confidence)} ({np.sum(high_confidence)/len(y_test)*100:.1f}%)")
        print(f"High confidence accuracy: {hc_accuracy * 100:.1f}%")

    # Save config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': NUM_FEATURES,
        'feature_names': FEATURE_NAMES,
        'min_move_threshold': MIN_MOVE_THRESHOLD,
        'test_accuracy': float(accuracy),
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
    print(f"FINAL DIRECTION ACCURACY: {accuracy * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
