"""
LSTM Model Training Script with GPU Support (NVIDIA CUDA)
Trains TensorFlow models for price prediction using your RTX 4080

Usage:
    python train_model_gpu.py <category> <asset> [epochs]

Examples:
    python train_model_gpu.py Crypto BTC 100
    python train_model_gpu.py "Stock Market" AAPL 50
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

# Enable GPU memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled")
else:
    print("No GPU found, using CPU (this will be slow)")

# Paths
DATA_DIR = Path(__file__).parent.parent / 'Data'
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
CONFIG = {
    'sequence_length': 30,    # 30 minutes lookback (predict based on last 30 min)
    'features': 5,            # OHLCV
    'lstm_units': [128, 64],  # Two LSTM layers
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'train_ratio': 0.8,
    'max_samples': 200000,    # GPU can handle more
    'batch_size': 256,        # Larger batch for GPU
}


def load_asset_data(category: str, asset: str):
    """Load candle data for an asset (historical + realtime buffer)"""
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

    # Also load from realtime buffer (recent/live data)
    buffer_file = DATA_DIR / '.buffer' / 'realtime.json'
    if buffer_file.exists():
        try:
            with open(buffer_file, 'r') as f:
                buffer_data = json.load(f)
                key = f'{category}/{asset}'
                if key in buffer_data and buffer_data[key].get('recentCandles'):
                    recent = buffer_data[key]['recentCandles']
                    all_candles.extend(recent)
                    print(f'Added {len(recent)} candles from realtime buffer')
        except (json.JSONDecodeError, IOError) as e:
            print(f'Warning: Could not load realtime buffer: {e}')

    # Remove duplicates and sort by timestamp
    unique = {c['timestamp']: c for c in all_candles}
    sorted_candles = sorted(unique.values(), key=lambda x: x['timestamp'])

    print(f'Historical: {historical_count:,} | Realtime: {len(sorted_candles) - historical_count:,} | Total: {len(sorted_candles):,}')

    return sorted_candles


def calculate_stats(candles):
    """Calculate normalization statistics"""
    prices = []
    volumes = []

    for c in candles:
        prices.extend([c['open'], c['high'], c['low'], c['close']])
        volumes.append(c['volume'])

    return {
        'min_price': min(prices),
        'max_price': max(prices),
        'min_volume': min(volumes),
        'max_volume': max(volumes),
    }


def normalize_candles(candles, stats):
    """Normalize candles to 0-1 range"""
    price_range = stats['max_price'] - stats['min_price'] or 1
    volume_range = stats['max_volume'] - stats['min_volume'] or 1

    normalized = []
    for c in candles:
        normalized.append({
            'open': (c['open'] - stats['min_price']) / price_range,
            'high': (c['high'] - stats['min_price']) / price_range,
            'low': (c['low'] - stats['min_price']) / price_range,
            'close': (c['close'] - stats['min_price']) / price_range,
            'volume': (c['volume'] - stats['min_volume']) / volume_range,
        })

    return normalized


def create_sequences(normalized_candles, sequence_length):
    """Create training sequences"""
    inputs = []
    outputs = []

    for i in range(sequence_length, len(normalized_candles)):
        seq = [[
            normalized_candles[j]['open'],
            normalized_candles[j]['high'],
            normalized_candles[j]['low'],
            normalized_candles[j]['close'],
            normalized_candles[j]['volume'],
        ] for j in range(i - sequence_length, i)]

        inputs.append(seq)
        outputs.append(normalized_candles[i]['close'])

    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)


def build_model():
    """Build LSTM model"""
    # Use implementation=1 for DirectML compatibility (non-cuDNN)
    model = tf.keras.Sequential([
        # First LSTM layer
        tf.keras.layers.LSTM(
            CONFIG['lstm_units'][0],
            return_sequences=True,
            input_shape=(CONFIG['sequence_length'], CONFIG['features']),
            implementation=1  # Force non-cuDNN implementation for DirectML
        ),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Second LSTM layer
        tf.keras.layers.LSTM(CONFIG['lstm_units'][1], return_sequences=False, implementation=1),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Dense layers
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_model(category: str, asset: str, epochs: int = 50):
    """Main training function"""
    print('=' * 60)
    print('S.U.P.I.D. LSTM Model Training (GPU)')
    print('=' * 60)
    print(f'Asset: {category}/{asset}')
    print(f'Epochs: {epochs}')
    print(f'Sequence Length: {CONFIG["sequence_length"]}')
    print(f'Batch Size: {CONFIG["batch_size"]}')
    print('=' * 60)
    print()

    # Check GPU
    print(f'TensorFlow version: {tf.__version__}')
    print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')
    print()

    # Load data
    print('Loading data...')
    candles = load_asset_data(category, asset)
    print(f'Loaded {len(candles):,} candles')

    if len(candles) < CONFIG['sequence_length'] + 100:
        raise ValueError(f"Not enough data. Need at least {CONFIG['sequence_length'] + 100} candles.")

    # Calculate stats
    stats = calculate_stats(candles)
    print(f'Price range: ${stats["min_price"]:.2f} - ${stats["max_price"]:.2f}')

    # Normalize
    print('Normalizing data...')
    normalized = normalize_candles(candles, stats)

    # Create sequences
    print('Creating training sequences...')
    inputs, outputs = create_sequences(normalized, CONFIG['sequence_length'])
    print(f'Created {len(inputs):,} sequences')

    # Sample if too many
    if len(inputs) > CONFIG['max_samples']:
        print(f'Sampling {CONFIG["max_samples"]:,} sequences (memory optimization)...')
        indices = np.linspace(0, len(inputs) - 1, CONFIG['max_samples'], dtype=int)
        inputs = inputs[indices]
        outputs = outputs[indices]
        print(f'Sampled to {len(inputs):,} sequences')

    # Split data
    split_idx = int(len(inputs) * CONFIG['train_ratio'])
    x_train, x_val = inputs[:split_idx], inputs[split_idx:]
    y_train, y_val = outputs[:split_idx], outputs[split_idx:]

    print(f'Training samples: {len(x_train):,}')
    print(f'Validation samples: {len(x_val):,}')
    print()

    # Build model
    print('Building model...')
    model = build_model()
    model.summary()

    print()
    print('=' * 60)
    print('Training started...')
    print('=' * 60)
    print()

    # Checkpoint directory for resume capability
    checkpoint_dir = MODELS_DIR / f'{asset.lower()}_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'epoch_{epoch:03d}.weights.h5'

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        ),
    ]

    # Check for existing checkpoint to resume from
    existing_checkpoints = sorted(checkpoint_dir.glob('epoch_*.weights.h5'))
    initial_epoch = 0
    if existing_checkpoints:
        latest = existing_checkpoints[-1]
        # Parse epoch number from filename like 'epoch_015.weights.h5'
        epoch_str = latest.stem.replace('.weights', '').split('_')[1]
        initial_epoch = int(epoch_str)
        print(f'Found checkpoint at epoch {initial_epoch}, resuming...')
        model.load_weights(str(latest))

    # Train
    start_time = datetime.now()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=CONFIG['batch_size'],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f'\nTraining completed in {elapsed:.1f} seconds')

    # Save model (Keras 3 requires .keras extension)
    model_name = f'{asset.lower()}_lstm'
    model_path = MODELS_DIR / f'{model_name}.keras'
    model.save(str(model_path))
    print(f'Model saved to: {model_path}')

    # Save stats for predictions
    stats_path = MODELS_DIR / f'{model_name}_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            **stats,
            'asset': asset,
            'category': category,
            'sequence_length': CONFIG['sequence_length'],
            'trained_at': datetime.now().isoformat(),
            'total_candles': len(candles),
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
        }, f, indent=2)
    print(f'Stats saved to: {stats_path}')

    print()
    print('=' * 60)
    print('Training Complete!')
    print(f'Final Loss: {history.history["loss"][-1]:.6f}')
    print(f'Final Val Loss: {history.history["val_loss"][-1]:.6f}')
    print('=' * 60)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python train_model_gpu.py <category> <asset> [epochs]')
        print()
        print('Examples:')
        print('  python train_model_gpu.py Crypto BTC 100')
        print('  python train_model_gpu.py "Stock Market" AAPL 50')
        sys.exit(1)

    category = sys.argv[1]
    asset = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    try:
        train_model(category, asset, epochs)
    except Exception as e:
        print(f'Training failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
