"""
LSTM Direction Classification Model v2
Targets >90% accuracy through:
1. 5-minute prediction horizon (not 1-minute noise)
2. Technical indicators (RSI, MACD, EMAs)
3. Only predicts SIGNIFICANT moves (>0.1% threshold)
4. Confidence thresholding at inference time
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
    'sequence_length': 30,       # 30 minutes of history (as requested)
    'prediction_horizon': 5,     # Predict 5 minutes ahead
    'move_threshold': 0.1,       # Only train on moves >0.1%
    'features': 15,              # Enhanced features with indicators
    'lstm_units': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.0001,
    'train_ratio': 0.8,
    'max_samples': 200000,       # More samples for overnight training
    'batch_size': 128,
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


def calculate_rsi(closes, period=14):
    """Calculate RSI indicator"""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))

    # First average
    if len(gains) >= period:
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Subsequent averages (exponential)
        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros(len(values))
    multiplier = 2 / (period + 1)
    ema[0] = values[0]

    for i in range(1, len(values)):
        ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))

    return ema


def calculate_macd(closes):
    """Calculate MACD indicator"""
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(closes, period=20, std_mult=2):
    """Calculate Bollinger Bands"""
    sma = np.zeros(len(closes))
    upper = np.zeros(len(closes))
    lower = np.zeros(len(closes))

    for i in range(period, len(closes)):
        window = closes[i-period:i]
        sma[i] = np.mean(window)
        std = np.std(window)
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std

    return sma, upper, lower


def calculate_features(candles):
    """Calculate enhanced features with technical indicators"""
    closes = np.array([c['close'] for c in candles])
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    opens = np.array([c['open'] for c in candles])
    volumes = np.array([c['volume'] for c in candles])

    # Technical indicators
    print('Calculating RSI...')
    rsi = calculate_rsi(closes)

    print('Calculating MACD...')
    macd_line, macd_signal, macd_hist = calculate_macd(closes)

    print('Calculating EMAs...')
    ema5 = calculate_ema(closes, 5)
    ema10 = calculate_ema(closes, 10)
    ema20 = calculate_ema(closes, 20)

    print('Calculating Bollinger Bands...')
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(closes)

    features_data = []
    horizon = CONFIG['prediction_horizon']
    threshold = CONFIG['move_threshold']

    # Start after we have enough data for indicators and can look ahead
    start_idx = max(30, CONFIG['sequence_length'])

    skipped_small = 0
    included = 0

    for i in range(start_idx, len(candles) - horizon):
        prev_close = candles[i - 1]['close']
        curr = candles[i]

        if prev_close <= 0:
            continue

        # Future price (5 minutes ahead)
        future_close = candles[i + horizon]['close']
        future_return = (future_close - curr['close']) / curr['close'] * 100

        # Only include SIGNIFICANT moves (>threshold%)
        if abs(future_return) < threshold:
            skipped_small += 1
            continue

        included += 1

        # Direction: 1 = UP, 0 = DOWN
        direction = 1 if future_return > 0 else 0

        # Basic returns
        close_return = (curr['close'] - prev_close) / prev_close * 100
        high_return = (curr['high'] - prev_close) / prev_close * 100
        low_return = (curr['low'] - prev_close) / prev_close * 100
        open_return = (curr['open'] - prev_close) / prev_close * 100

        # Volume
        log_volume = np.log1p(curr['volume']) if curr['volume'] > 0 else 0

        # Price patterns
        price_range = (curr['high'] - curr['low']) / prev_close * 100
        body = (curr['close'] - curr['open']) / prev_close * 100
        upper_wick = (curr['high'] - max(curr['open'], curr['close'])) / prev_close * 100
        lower_wick = (min(curr['open'], curr['close']) - curr['low']) / prev_close * 100

        # Technical indicator values (normalized)
        rsi_val = rsi[i] / 100  # 0-1 scale

        # MACD (normalize by price)
        macd_val = macd_hist[i] / curr['close'] * 100

        # EMA distances (how far price is from EMAs)
        ema5_dist = (curr['close'] - ema5[i]) / curr['close'] * 100
        ema10_dist = (curr['close'] - ema10[i]) / curr['close'] * 100
        ema20_dist = (curr['close'] - ema20[i]) / curr['close'] * 100

        # Bollinger position (where in the bands)
        bb_range = bb_upper[i] - bb_lower[i] if bb_upper[i] > bb_lower[i] else 1
        bb_position = (curr['close'] - bb_lower[i]) / bb_range  # 0-1 scale

        features_data.append({
            # Basic price features (5)
            'close_ret': np.clip(close_return / 5, -1, 1),
            'high_ret': np.clip(high_return / 5, -1, 1),
            'low_ret': np.clip(low_return / 5, -1, 1),
            'open_ret': np.clip(open_return / 5, -1, 1),
            'log_volume': log_volume,

            # Candle patterns (3)
            'price_range': np.clip(price_range / 5, 0, 1),
            'body': np.clip(body / 5, -1, 1),
            'wick_ratio': np.clip((upper_wick - lower_wick) / 5, -1, 1),

            # Technical indicators (7)
            'rsi': np.clip(rsi_val, 0, 1),
            'macd': np.clip(macd_val / 2, -1, 1),
            'ema5_dist': np.clip(ema5_dist / 2, -1, 1),
            'ema10_dist': np.clip(ema10_dist / 2, -1, 1),
            'ema20_dist': np.clip(ema20_dist / 2, -1, 1),
            'bb_position': np.clip(bb_position, 0, 1),
            'trend': np.clip((ema5[i] - ema20[i]) / curr['close'] * 100, -1, 1),

            # Target
            'direction': direction,
            'actual_return': future_return,
        })

    print(f'Skipped {skipped_small:,} small moves (<{threshold}%)')
    print(f'Included {included:,} significant moves (>{threshold}%)')

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

    feature_keys = [
        'close_ret', 'high_ret', 'low_ret', 'open_ret', 'log_volume',
        'price_range', 'body', 'wick_ratio',
        'rsi', 'macd', 'ema5_dist', 'ema10_dist', 'ema20_dist', 'bb_position', 'trend'
    ]

    for i in range(sequence_length, len(features_data)):
        seq = [[features_data[j][k] for k in feature_keys]
               for j in range(i - sequence_length, i)]

        inputs.append(seq)
        outputs.append(features_data[i]['direction'])

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
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(category: str, asset: str, epochs: int = 200):
    """Main training function"""
    print('=' * 70)
    print('LSTM Direction Classification Model v2')
    print('=' * 70)
    print(f'Asset: {category}/{asset}')
    print(f'Epochs: {epochs}')
    print(f'Prediction horizon: {CONFIG["prediction_horizon"]} minutes')
    print(f'Move threshold: >{CONFIG["move_threshold"]}%')
    print(f'Features: {CONFIG["features"]} (with technical indicators)')
    print(f'Target: >90% direction accuracy')
    print('=' * 70)
    print()

    print('Loading data...')
    candles = load_asset_data(category, asset)

    if len(candles) < CONFIG['sequence_length'] + 500:
        raise ValueError("Not enough data")

    print('Calculating features with technical indicators...')
    features_data = calculate_features(candles)
    print(f'Generated {len(features_data):,} samples (significant moves only)')

    if len(features_data) < 1000:
        raise ValueError("Not enough significant moves in data")

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

    # Calculate class weights
    pos_weight = len(y_train) / (2 * max(np.sum(y_train), 1))
    neg_weight = len(y_train) / (2 * max(len(y_train) - np.sum(y_train), 1))
    class_weights = {0: neg_weight, 1: pos_weight}
    print(f'Class weights: DOWN={neg_weight:.2f}, UP={pos_weight:.2f}')
    print()

    print('Building model...')
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=15,
            min_lr=0.000001,
            mode='max',
            verbose=1
        ),
    ]

    print()
    print('=' * 70)
    print('Training for >90% accuracy...')
    print('=' * 70)

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
    best_accuracy = max(history.history['val_accuracy']) * 100

    print(f'\nFinal Validation Accuracy: {val_accuracy*100:.1f}%')
    print(f'Best Validation Accuracy: {best_accuracy:.1f}%')

    # Save model
    model_name = f'{asset.lower()}_lstm_direction_v2'
    model_path = MODELS_DIR / f'{model_name}.keras'
    model.save(str(model_path))
    print(f'Model saved: {model_path}')

    # Save stats
    stats = {
        'asset': asset,
        'category': category,
        'model_type': 'direction_v2',
        'sequence_length': CONFIG['sequence_length'],
        'prediction_horizon': CONFIG['prediction_horizon'],
        'move_threshold': CONFIG['move_threshold'],
        'max_volume': max_volume,
        'features': CONFIG['features'],
        'trained_at': datetime.now().isoformat(),
        'total_candles': len(candles),
        'significant_moves': len(features_data),
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
    print('=' * 70)
    print('Training Complete!')
    print(f'Best Validation Accuracy: {best_accuracy:.1f}%')
    if best_accuracy >= 90:
        print('[SUCCESS] Target >90% achieved!')
    elif best_accuracy >= 80:
        print('[GOOD] >80% achieved, close to target')
    elif best_accuracy >= 70:
        print('[PROGRESS] >70% achieved, improving')
    else:
        print('[NEEDS WORK] <70%, may need more data or features')
    print('=' * 70)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python train_model_direction_v2.py <category> <asset> [epochs]')
        sys.exit(1)

    category = sys.argv[1]
    asset = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    try:
        train_model(category, asset, epochs)
    except Exception as e:
        print(f'Training failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
