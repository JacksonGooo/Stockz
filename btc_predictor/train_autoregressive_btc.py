"""
Autoregressive LSTM Model for BTC Price Prediction
Trains model to predict next 1-minute candle, then generates 60-minute forecasts

Based on insights:
- Patterns are mathematical (order book structure), not psychology
- GPT-style: predict one candle, feed back, repeat
- Shorter timeframes = more predictable
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import json

# GPU enabled for WSL2 with CUDA
# tf.config.set_visible_devices([], 'GPU')  # Commented out - use GPU!

from data_loader import load_and_prepare_data, SEQUENCE_LENGTH, PREDICTION_HORIZON, get_random_test_samples, load_all_btc_candles
from features import NUM_FEATURES, FEATURE_NAMES

# Model configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'autoregressive_btc')
BATCH_SIZE = 512  # Larger batch for GPU efficiency (RTX 4050 6GB)
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping patience


def build_model(sequence_length, num_features, output_size=5):
    """
    Build the autoregressive LSTM model

    Architecture:
    - Input: (sequence_length, num_features)
    - LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> Dense(32) -> Dense(5)
    - Output: Next candle's OHLCV returns
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, num_features)),

        # First LSTM layer - captures longer patterns
        # Note: No recurrent_dropout - allows cuDNN acceleration on GPU
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        # Second LSTM layer - refines predictions
        layers.LSTM(64, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Dense layers for prediction
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        # Output: 5 values (open_return, high_return, low_return, close_return, volume_change)
        layers.Dense(output_size, activation='linear')
    ])

    return model


def train_model(X_train, y_train, X_test, y_test):
    """Train the autoregressive model"""

    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = build_model(SEQUENCE_LENGTH, NUM_FEATURES)

    # Compile with MSE loss (we're predicting continuous returns)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    model.summary()

    # Callbacks
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_test):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {EPOCHS}")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def predict_60_minutes(model, initial_sequence, feature_indices):
    """
    Autoregressive prediction: predict 60 candles by feeding predictions back

    Args:
        model: Trained LSTM model
        initial_sequence: (1, sequence_length, num_features) input
        feature_indices: Indices of OHLCV return features in the feature vector

    Returns:
        List of 60 predicted candle returns
    """
    predictions = []
    current_seq = initial_sequence.copy()

    # Get indices for OHLCV return features
    ohlcv_indices = feature_indices

    for i in range(PREDICTION_HORIZON):
        # Predict next candle
        pred = model.predict(current_seq, verbose=0)[0]
        predictions.append(pred)

        # Create new feature vector for predicted candle
        new_features = current_seq[0, -1, :].copy()  # Start with last candle's features

        # Update OHLCV returns with prediction
        for j, idx in enumerate(ohlcv_indices):
            new_features[idx] = pred[j]

        # Shift sequence and append prediction
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_features

    return np.array(predictions)


def evaluate_model(model, df):
    """
    Evaluate model by testing on random timestamps and comparing to actual
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL - Random Timestamp Tests")
    print("=" * 60)

    # Get random test samples
    samples = get_random_test_samples(df, n_samples=10)

    if not samples:
        print("Not enough data for evaluation!")
        return

    # Feature indices for OHLCV returns
    ohlcv_features = ['open_return', 'high_return', 'low_return', 'close_return', 'volume_change']
    feature_indices = [FEATURE_NAMES.index(f) for f in ohlcv_features]

    results = []

    for i, sample in enumerate(samples):
        print(f"\nTest {i+1}: {sample['start_datetime']}")

        # Get input sequence
        input_seq = sample['input_sequence'][np.newaxis, :, :]  # Add batch dimension
        actual_candles = sample['actual_candles']

        # Predict 60 minutes
        predictions = predict_60_minutes(model, input_seq, feature_indices)

        # Extract actual OHLCV returns
        actual_returns = actual_candles[:, feature_indices]

        # Calculate cumulative returns (price change over 60 minutes)
        predicted_cum_return = np.sum(predictions[:, 3])  # Sum of close returns
        actual_cum_return = np.sum(actual_returns[:, 3])

        # Predicted direction
        pred_direction = "UP" if predicted_cum_return > 0 else "DOWN"
        actual_direction = "UP" if actual_cum_return > 0 else "DOWN"
        correct = pred_direction == actual_direction

        # MSE for next-candle prediction
        next_candle_mse = np.mean((predictions[0] - actual_returns[0]) ** 2)

        # MAE for cumulative return
        cum_return_mae = abs(predicted_cum_return - actual_cum_return)

        print(f"  Predicted 60-min return: {predicted_cum_return*100:.4f}%")
        print(f"  Actual 60-min return:    {actual_cum_return*100:.4f}%")
        print(f"  Direction: Predicted {pred_direction}, Actual {actual_direction} {'[CORRECT]' if correct else '[WRONG]'}")
        print(f"  Next-candle MSE: {next_candle_mse:.8f}")

        results.append({
            'timestamp': str(sample['start_datetime']),
            'predicted_return': float(predicted_cum_return),
            'actual_return': float(actual_cum_return),
            'direction_correct': correct,
            'next_candle_mse': float(next_candle_mse),
            'cum_return_mae': float(cum_return_mae)
        })

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    correct_count = sum(1 for r in results if r['direction_correct'])
    direction_accuracy = correct_count / len(results) * 100
    avg_next_candle_mse = np.mean([r['next_candle_mse'] for r in results])
    avg_cum_mae = np.mean([r['cum_return_mae'] for r in results])

    print(f"Direction Accuracy: {direction_accuracy:.1f}% ({correct_count}/{len(results)})")
    print(f"Avg Next-Candle MSE: {avg_next_candle_mse:.8f}")
    print(f"Avg Cumulative Return MAE: {avg_cum_mae*100:.4f}%")

    return results


def save_model(model, history, evaluation_results=None):
    """Save model weights and training history"""

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model weights
    model_path = os.path.join(MODEL_DIR, 'model.keras')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training history
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'epochs': len(history.history['loss']),
        'trained_at': datetime.now().isoformat()
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Save evaluation results
    if evaluation_results:
        eval_path = os.path.join(MODEL_DIR, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation results saved to: {eval_path}")

    # Save model configuration
    config_path = os.path.join(MODEL_DIR, 'config.json')
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': NUM_FEATURES,
        'feature_names': FEATURE_NAMES,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'architecture': 'LSTM(128)->LSTM(64)->Dense(32)->Dense(5)'
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to: {config_path}")


def main():
    print("=" * 60)
    print("AUTOREGRESSIVE BTC PRICE PREDICTION")
    print("=" * 60)
    print(f"Sequence length: {SEQUENCE_LENGTH} minutes (lookback)")
    print(f"Prediction horizon: {PREDICTION_HORIZON} minutes")
    print(f"Features: {NUM_FEATURES}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load and prepare data
    print("\nLoading data...")
    X_train, y_train, X_test, y_test, df = load_and_prepare_data()

    if X_train is None:
        print("Failed to load data!")
        return

    print(f"\nData loaded successfully!")
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test)

    # Evaluate on random timestamps
    evaluation_results = evaluate_model(model, df)

    # Save everything
    save_model(model, history, evaluation_results)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
