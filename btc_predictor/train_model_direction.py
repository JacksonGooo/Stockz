"""
LSTM Direction Classification Model
Predicts UP or DOWN (binary classification) instead of price/returns

This model is optimized for direction accuracy, not price accuracy.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

DATA_DIR = Path(__file__).parent.parent / 'Data'
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

CONFIG = {
    'sequence_length': 30,
    'features': 8,  # More features for direction
    'lstm_units': [128, 64, 32],  # Deeper network
    'dropout_rate': 0.4,  # Higher dropout to prevent overfitting
    'learning_rate': 0.0003,
    'train_ratio': 0.8,
    'max_samples': 200000,
    'batch_size': 256,
}


def load_asset_data(category: str, asset: str):
    """Load candle data for an asset"""
    asset_dir = DATA_DIR / category / asset

    if not asset_dir.exists():
        raise FileNotFoundError(f"No data found for {category}/{asset}")

    all_candles = []
    week_dirs = sorted([d for d in asset_dir.iterdir() if d.name.startswith('week_')])

    for week_dir in week_dirs:
        json_files = sorted(week_dir.glob('*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_candles.extend(data)
            except (json.JSONDecodeError, IOError):
                continue

    unique = {c['timestamp']: c for c in all_candles}
    sorted_candles = sorted(unique.values(), key=lambda x: x['timestamp'])

    print(f'Loaded {len(sorted_candles):,} candles')
    return sorted_candles


def calculate_features(candles):
    """Calculate enhanced features for direction prediction"""
    features_data = []

    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]

        if prev['close'] <= 0:
            continue

        # Basic returns
        close_return = (curr['close'] - prev['close']) / prev['close'] * 100
        high_return = (curr['high'] - prev['close']) / prev['close'] * 100
        low_return = (curr['low'] - prev['close']) / prev['close'] * 100
        open_return = (curr['open'] - prev['close']) / prev['close'] * 100

        # Volume features
        log_volume = np.log1p(curr['volume']) if curr['volume'] > 0 else 0

        # Volatility (price range)
        price_range = (curr['high'] - curr['low']) / prev['close'] * 100

        # Candle body (bullish/bearish strength)
        body = (curr['close'] - curr['open']) / prev['close'] * 100

        # Upper/lower wicks (rejection signals)
        upper_wick = (curr['high'] - max(curr['open'], curr['close'])) / prev['close'] * 100
        lower_wick = (min(curr['open'], curr['close']) - curr['low']) / prev['close'] * 100

        # Direction label: 1 = UP, 0 = DOWN
        direction = 1 if close_return > 0 else 0

        features_data.append({
            'open_ret': np.clip(open_return / 5, -1, 1),
            'high_ret': np.clip(high_return / 5, -1, 1),
            'low_ret': np.clip(low_return / 5, -1, 1),
            'close_ret': np.clip(close_return / 5, -1, 1),
            'log_volume': log_volume,
            'price_range': np.clip(price_range / 5, 0, 1),
            'body': np.clip(body / 5, -1, 1),
            'wick_ratio': np.clip((upper_wick - lower_wick) / 5, -1, 1),
            'direction': direction,
            'actual_return': close_return,
        })

    return features_data


def normalize_features(features_data):
    """Normalize volume feature"""
    all_volumes = [f['log_volume'] for f in features_data]
    max_vol = max(all_volumes) if all_volumes else 1

    for f in features_data:
        f['log_volume'] = f['log_volume'] / max_vol if max_vol > 0 else 0

    return features_data, max_vol


def create_sequences(features_data, sequence_length):
    """Create training sequences for direction classification"""
    inputs = []
    outputs = []

    for i in range(sequence_length, len(features_data)):
        seq = [[
            features_data[j]['open_ret'],
            features_data[j]['high_ret'],
            features_data[j]['low_ret'],
            features_data[j]['close_ret'],
            features_data[j]['log_volume'],
            features_data[j]['price_range'],
            features_data[j]['body'],
            features_data[j]['wick_ratio'],
        ] for j in range(i - sequence_length, i)]

        inputs.append(seq)
        outputs.append(features_data[i]['direction'])  # Binary: 0 or 1

    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)


def build_model():
    """Build LSTM model for binary direction classification"""
    model = tf.keras.Sequential([
        # First LSTM layer
        tf.keras.layers.LSTM(
            CONFIG['lstm_units'][0],
            return_sequences=True,
            input_shape=(CONFIG['sequence_length'], CONFIG['features']),
            implementation=1
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Second LSTM layer
        tf.keras.layers.LSTM(CONFIG['lstm_units'][1], return_sequences=True, implementation=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Third LSTM layer
        tf.keras.layers.LSTM(CONFIG['lstm_units'][2], return_sequences=False, implementation=1),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Dense layers
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),  # Binary output: 0-1
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='binary_crossentropy',  # Classification loss
        metrics=['accuracy']  # Track accuracy directly
    )

    return model


def train_model(category: str, asset: str, epochs: int = 100):
    """Main training function"""
    print('=' * 60)
    print('LSTM Direction Classification Model')
    print('=' * 60)
    print(f'Asset: {category}/{asset}')
    print(f'Epochs: {epochs}')
    print(f'Predicts: UP (1) or DOWN (0)')
    print('=' * 60)
    print()

    print('Loading data...')
    candles = load_asset_data(category, asset)

    if len(candles) < CONFIG['sequence_length'] + 100:
        raise ValueError("Not enough data")

    print('Calculating features...')
    features_data = calculate_features(candles)
    print(f'Generated {len(features_data):,} samples')

    # Check class balance
    up_count = sum(1 for f in features_data if f['direction'] == 1)
    down_count = len(features_data) - up_count
    print(f'Class balance: UP={up_count:,} ({100*up_count/len(features_data):.1f}%) | DOWN={down_count:,} ({100*down_count/len(features_data):.1f}%)')

    print('Normalizing...')
    features_data, max_volume = normalize_features(features_data)

    print('Creating sequences...')
    inputs, outputs = create_sequences(features_data, CONFIG['sequence_length'])
    print(f'Created {len(inputs):,} sequences')

    # Sample if too many
    if len(inputs) > CONFIG['max_samples']:
        print(f'Sampling {CONFIG["max_samples"]:,} sequences...')
        indices = np.random.choice(len(inputs), CONFIG['max_samples'], replace=False)
        indices = np.sort(indices)
        inputs = inputs[indices]
        outputs = outputs[indices]

    # Split data
    split_idx = int(len(inputs) * CONFIG['train_ratio'])
    x_train, x_val = inputs[:split_idx], inputs[split_idx:]
    y_train, y_val = outputs[:split_idx], outputs[split_idx:]

    print(f'Training: {len(x_train):,} | Validation: {len(x_val):,}')

    # Calculate class weights to handle imbalance
    pos_weight = len(y_train) / (2 * np.sum(y_train))
    neg_weight = len(y_train) / (2 * (len(y_train) - np.sum(y_train)))
    class_weights = {0: neg_weight, 1: pos_weight}
    print(f'Class weights: DOWN={neg_weight:.2f}, UP={pos_weight:.2f}')
    print()

    print('Building model...')
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitor accuracy for classification
            patience=20,
            restore_best_weights=True,
            mode='max',  # Maximize accuracy
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            mode='max',
            verbose=1
        ),
    ]

    print()
    print('=' * 60)
    print('Training...')
    print('=' * 60)

    start_time = datetime.now()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=CONFIG['batch_size'],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f'\nTraining completed in {elapsed:.1f}s')

    # Evaluate
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f'\nFinal Validation Accuracy: {val_accuracy*100:.1f}%')

    # Save model
    model_name = f'{asset.lower()}_lstm_direction'
    model_path = MODELS_DIR / f'{model_name}.keras'
    model.save(str(model_path))
    print(f'Model saved: {model_path}')

    # Save stats
    stats = {
        'asset': asset,
        'category': category,
        'model_type': 'direction',
        'sequence_length': CONFIG['sequence_length'],
        'max_volume': max_volume,
        'features': CONFIG['features'],
        'trained_at': datetime.now().isoformat(),
        'total_candles': len(candles),
        'epochs_trained': len(history.history['loss']),
        'final_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
    }

    stats_path = MODELS_DIR / f'{model_name}_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats saved: {stats_path}')

    print()
    print('=' * 60)
    print('Training Complete!')
    print(f'Best Validation Accuracy: {max(history.history["val_accuracy"])*100:.1f}%')
    print('=' * 60)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python train_model_direction.py <category> <asset> [epochs]')
        sys.exit(1)

    category = sys.argv[1]
    asset = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    try:
        train_model(category, asset, epochs)
    except Exception as e:
        print(f'Training failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
