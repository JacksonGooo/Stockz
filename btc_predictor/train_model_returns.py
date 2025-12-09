"""
LSTM Model Training Script - Returns-Based
Predicts percentage price change instead of raw prices (much better for ML)

Usage:
    python train_model_returns.py <category> <asset> [epochs]

Examples:
    python train_model_returns.py Crypto BTC 100
    python train_model_returns.py "Stock Market" AAPL 50
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Paths
DATA_DIR = Path(__file__).parent.parent / 'Data'
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
CONFIG = {
    'sequence_length': 30,    # 30 minutes lookback
    'features': 6,            # OHLCV + return
    'lstm_units': [64, 32],   # Smaller model for returns (simpler task)
    'dropout_rate': 0.3,      # Higher dropout for regularization
    'learning_rate': 0.0005,  # Lower learning rate
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

    # Load historical data from week folders
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

    historical_count = len(all_candles)

    # Also load from realtime buffer
    buffer_file = DATA_DIR / '.buffer' / 'realtime.json'
    if buffer_file.exists():
        try:
            with open(buffer_file, 'r') as f:
                buffer_data = json.load(f)
                key = f'{category}/{asset}'
                if key in buffer_data and buffer_data[key].get('recentCandles'):
                    recent = buffer_data[key]['recentCandles']
                    all_candles.extend(recent)
        except (json.JSONDecodeError, IOError):
            pass

    # Remove duplicates and sort
    unique = {c['timestamp']: c for c in all_candles}
    sorted_candles = sorted(unique.values(), key=lambda x: x['timestamp'])

    print(f'Loaded {len(sorted_candles):,} candles ({historical_count:,} historical)')

    return sorted_candles


def calculate_returns(candles):
    """Calculate percentage returns for each candle"""
    returns_data = []

    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]

        # Calculate returns as percentage change
        if prev['close'] > 0:
            close_return = (curr['close'] - prev['close']) / prev['close'] * 100
            high_return = (curr['high'] - prev['close']) / prev['close'] * 100
            low_return = (curr['low'] - prev['close']) / prev['close'] * 100
            open_return = (curr['open'] - prev['close']) / prev['close'] * 100

            # Normalize volume (log scale)
            log_volume = np.log1p(curr['volume']) if curr['volume'] > 0 else 0

            # Price range (volatility indicator)
            price_range = (curr['high'] - curr['low']) / prev['close'] * 100

            returns_data.append({
                'timestamp': curr['timestamp'],
                'open_ret': open_return,
                'high_ret': high_return,
                'low_ret': low_return,
                'close_ret': close_return,
                'log_volume': log_volume,
                'price_range': price_range,
                'actual_close': curr['close'],
                'prev_close': prev['close'],
            })

    return returns_data


def normalize_returns(returns_data):
    """Normalize returns to roughly -1 to 1 range"""
    # Returns are typically -5% to +5% per minute, so divide by 5
    # Volume is log-scaled, normalize to 0-1
    all_volumes = [r['log_volume'] for r in returns_data]
    max_vol = max(all_volumes) if all_volumes else 1

    normalized = []
    for r in returns_data:
        normalized.append({
            'open_ret': np.clip(r['open_ret'] / 5, -1, 1),
            'high_ret': np.clip(r['high_ret'] / 5, -1, 1),
            'low_ret': np.clip(r['low_ret'] / 5, -1, 1),
            'close_ret': np.clip(r['close_ret'] / 5, -1, 1),
            'log_volume': r['log_volume'] / max_vol if max_vol > 0 else 0,
            'price_range': np.clip(r['price_range'] / 5, 0, 1),
            'target_return': r['close_ret'],  # Keep unnormalized for output
            'actual_close': r['actual_close'],
            'prev_close': r['prev_close'],
        })

    return normalized, max_vol


def create_sequences(normalized_data, sequence_length):
    """Create training sequences"""
    inputs = []
    outputs = []
    metadata = []  # Store prev_close for de-normalization

    for i in range(sequence_length, len(normalized_data)):
        # Input: sequence of normalized features
        seq = [[
            normalized_data[j]['open_ret'],
            normalized_data[j]['high_ret'],
            normalized_data[j]['low_ret'],
            normalized_data[j]['close_ret'],
            normalized_data[j]['log_volume'],
            normalized_data[j]['price_range'],
        ] for j in range(i - sequence_length, i)]

        inputs.append(seq)

        # Output: next return (normalized to -1 to 1)
        target = np.clip(normalized_data[i]['target_return'] / 5, -1, 1)
        outputs.append(target)

        # Metadata for testing
        metadata.append({
            'prev_close': normalized_data[i]['prev_close'],
            'actual_close': normalized_data[i]['actual_close'],
            'actual_return': normalized_data[i]['target_return'],
        })

    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32), metadata


def build_model():
    """Build LSTM model for return prediction"""
    model = tf.keras.Sequential([
        # First LSTM layer
        tf.keras.layers.LSTM(
            CONFIG['lstm_units'][0],
            return_sequences=True,
            input_shape=(CONFIG['sequence_length'], CONFIG['features']),
            implementation=1
        ),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Second LSTM layer
        tf.keras.layers.LSTM(CONFIG['lstm_units'][1], return_sequences=False, implementation=1),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Dense layers
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh'),  # Output -1 to 1
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_model(category: str, asset: str, epochs: int = 100):
    """Main training function"""
    print('=' * 60)
    print('LSTM Returns Model Training')
    print('=' * 60)
    print(f'Asset: {category}/{asset}')
    print(f'Epochs: {epochs}')
    print(f'Predicts: Percentage return (next minute)')
    print('=' * 60)
    print()

    # Load data
    print('Loading data...')
    candles = load_asset_data(category, asset)

    if len(candles) < CONFIG['sequence_length'] + 100:
        raise ValueError(f"Not enough data.")

    # Calculate returns
    print('Calculating returns...')
    returns_data = calculate_returns(candles)
    print(f'Generated {len(returns_data):,} return samples')

    # Get stats for reference
    all_returns = [r['close_ret'] for r in returns_data]
    print(f'Return range: {min(all_returns):.2f}% to {max(all_returns):.2f}%')
    print(f'Mean return: {np.mean(all_returns):.4f}%')
    print(f'Std return: {np.std(all_returns):.4f}%')

    # Normalize
    print('Normalizing...')
    normalized, max_volume = normalize_returns(returns_data)

    # Create sequences
    print('Creating sequences...')
    inputs, outputs, metadata = create_sequences(normalized, CONFIG['sequence_length'])
    print(f'Created {len(inputs):,} sequences')

    # Sample if too many
    if len(inputs) > CONFIG['max_samples']:
        print(f'Sampling {CONFIG["max_samples"]:,} sequences...')
        indices = np.linspace(0, len(inputs) - 1, CONFIG['max_samples'], dtype=int)
        inputs = inputs[indices]
        outputs = outputs[indices]

    # Split data
    split_idx = int(len(inputs) * CONFIG['train_ratio'])
    x_train, x_val = inputs[:split_idx], inputs[split_idx:]
    y_train, y_val = outputs[:split_idx], outputs[split_idx:]

    print(f'Training: {len(x_train):,} | Validation: {len(x_val):,}')
    print()

    # Build model
    print('Building model...')
    model = build_model()
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # More patience for returns
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
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
        verbose=1
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f'\nTraining completed in {elapsed:.1f}s')

    # Save model
    model_name = f'{asset.lower()}_lstm_returns'
    model_path = MODELS_DIR / f'{model_name}.keras'
    model.save(str(model_path))
    print(f'Model saved: {model_path}')

    # Save stats
    stats = {
        'asset': asset,
        'category': category,
        'model_type': 'returns',
        'sequence_length': CONFIG['sequence_length'],
        'max_volume': max_volume,
        'return_scale': 5,  # Returns are divided by this
        'trained_at': datetime.now().isoformat(),
        'total_candles': len(candles),
        'epochs_trained': len(history.history['loss']),
        'final_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'return_stats': {
            'mean': float(np.mean(all_returns)),
            'std': float(np.std(all_returns)),
            'min': float(min(all_returns)),
            'max': float(max(all_returns)),
        }
    }

    stats_path = MODELS_DIR / f'{model_name}_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats saved: {stats_path}')

    print()
    print('=' * 60)
    print('Training Complete!')
    print(f'Final Loss: {history.history["loss"][-1]:.6f}')
    print(f'Final Val Loss: {history.history["val_loss"][-1]:.6f}')
    print('=' * 60)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python train_model_returns.py <category> <asset> [epochs]')
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
