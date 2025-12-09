"""
Enhanced LSTM Model Training Script with 20+ Features
Trains TensorFlow models for price prediction using your RTX 4080

Features included:
- Price-Based: OHLC, returns, log returns
- Volatility: Rolling std, ATR, high-low spread
- Volume: Raw, relative volume
- Technical: SMA, RSI, MACD, Bollinger Bands
- Time Context: Hour of day, day of week (sin/cos encoded)

Usage:
    python train_model_enhanced.py <category> <asset> [epochs]

Examples:
    python train_model_enhanced.py Crypto BTC 100
    python train_model_enhanced.py "Stock Market" AAPL 50
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Model configuration - ENHANCED with more features
CONFIG = {
    'sequence_length': 60,    # 60 minutes lookback
    'features': 25,           # ENHANCED: 25 features now
    'lstm_units': [256, 128, 64],  # Three LSTM layers for more capacity
    'dropout_rate': 0.3,
    'learning_rate': 0.0005,  # Lower LR for stability with more features
    'train_ratio': 0.85,
    'max_samples': 300000,    # GPU can handle more
    'batch_size': 256,
}

# =============================================================================
# Technical Indicator Calculations
# =============================================================================

def calculate_sma(prices, period):
    """Simple Moving Average"""
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1:i + 1])
    return result


def calculate_ema(prices, period):
    """Exponential Moving Average"""
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    result = np.full(len(prices), np.nan)
    multiplier = 2 / (period + 1)

    # Start with SMA for initial value
    result[period - 1] = np.mean(prices[:period])

    for i in range(period, len(prices)):
        result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]

    return result


def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return np.full(len(prices), 50)  # Neutral

    result = np.full(len(prices), np.nan)

    # Calculate price changes
    deltas = np.diff(prices)

    for i in range(period, len(prices)):
        gains = []
        losses = []
        for j in range(i - period, i):
            delta = deltas[j] if j < len(deltas) else 0
            if delta > 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss == 0:
            result[i] = 100
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line[~np.isnan(macd_line)], signal)

    # Pad signal line
    signal_padded = np.full(len(prices), np.nan)
    valid_start = np.argmax(~np.isnan(macd_line))
    if len(signal_line) > 0:
        signal_padded[valid_start + signal - 1:valid_start + signal - 1 + len(signal_line)] = signal_line

    histogram = macd_line - signal_padded

    return macd_line, signal_padded, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands"""
    sma = calculate_sma(prices, period)

    # Calculate rolling std
    std = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1:i + 1])

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    # Position within bands (0 = at lower, 1 = at upper)
    band_width = upper - lower
    band_width[band_width == 0] = 1  # Avoid division by zero
    position = (prices - lower) / band_width

    return upper, lower, position


def calculate_atr(highs, lows, closes, period=14):
    """Average True Range"""
    if len(closes) < 2:
        return np.full(len(closes), np.nan)

    # True Range
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]

    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR (EMA of TR)
    atr = calculate_ema(tr, period)
    return atr


def calculate_rolling_volatility(returns, period=20):
    """Rolling standard deviation of returns"""
    if len(returns) < period:
        return np.full(len(returns), np.nan)

    result = np.full(len(returns), np.nan)
    for i in range(period - 1, len(returns)):
        result[i] = np.std(returns[i - period + 1:i + 1])
    return result


# =============================================================================
# Data Loading and Feature Engineering
# =============================================================================

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


def engineer_features(candles):
    """Calculate all 25 features from raw OHLCV data"""
    n = len(candles)

    # Extract base data
    timestamps = np.array([c['timestamp'] for c in candles])
    opens = np.array([c['open'] for c in candles], dtype=np.float64)
    highs = np.array([c['high'] for c in candles], dtype=np.float64)
    lows = np.array([c['low'] for c in candles], dtype=np.float64)
    closes = np.array([c['close'] for c in candles], dtype=np.float64)
    volumes = np.array([c['volume'] for c in candles], dtype=np.float64)

    print("Calculating features...")

    # === Price-Based Features ===
    # 1-4: OHLC (normalized later)

    # 5: Price returns (percent change)
    returns = np.zeros(n)
    returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    returns = np.nan_to_num(returns, nan=0, posinf=0, neginf=0)

    # 6: Log returns
    log_returns = np.zeros(n)
    log_returns[1:] = np.log(closes[1:] / closes[:-1])
    log_returns = np.nan_to_num(log_returns, nan=0, posinf=0, neginf=0)

    # === Volatility Features ===
    # 7: High-Low spread (intra-bar volatility)
    hl_spread = (highs - lows) / closes
    hl_spread = np.nan_to_num(hl_spread, nan=0)

    # 8: Rolling volatility (5-min)
    vol_5 = calculate_rolling_volatility(returns, 5)
    vol_5 = np.nan_to_num(vol_5, nan=0)

    # 9: Rolling volatility (20-min)
    vol_20 = calculate_rolling_volatility(returns, 20)
    vol_20 = np.nan_to_num(vol_20, nan=0)

    # 10: ATR (14-period)
    atr = calculate_atr(highs, lows, closes, 14)
    atr = np.nan_to_num(atr, nan=0)

    # === Volume Features ===
    # 11: Raw volume (normalized later)

    # 12: Relative volume (vs 20-period avg)
    vol_sma = calculate_sma(volumes, 20)
    vol_sma[vol_sma == 0] = 1
    rel_volume = volumes / vol_sma
    rel_volume = np.nan_to_num(rel_volume, nan=1, posinf=1)

    # === Technical Indicators ===
    # 13: SMA 5
    sma_5 = calculate_sma(closes, 5)
    sma_5_pct = (closes - sma_5) / sma_5
    sma_5_pct = np.nan_to_num(sma_5_pct, nan=0)

    # 14: SMA 20
    sma_20 = calculate_sma(closes, 20)
    sma_20_pct = (closes - sma_20) / sma_20
    sma_20_pct = np.nan_to_num(sma_20_pct, nan=0)

    # 15: SMA 50
    sma_50 = calculate_sma(closes, 50)
    sma_50_pct = (closes - sma_50) / sma_50
    sma_50_pct = np.nan_to_num(sma_50_pct, nan=0)

    # 16: RSI (14-period) - already 0-100, normalize to 0-1
    rsi = calculate_rsi(closes, 14)
    rsi = np.nan_to_num(rsi, nan=50) / 100

    # 17-19: MACD (line, signal, histogram)
    macd_line, macd_signal, macd_hist = calculate_macd(closes)
    # Normalize MACD by price
    macd_line_norm = np.nan_to_num(macd_line / closes, nan=0)
    macd_signal_norm = np.nan_to_num(macd_signal / closes, nan=0)
    macd_hist_norm = np.nan_to_num(macd_hist / closes, nan=0)

    # 20: Bollinger Band position (0-1 scale)
    _, _, bb_position = calculate_bollinger_bands(closes, 20, 2)
    bb_position = np.clip(np.nan_to_num(bb_position, nan=0.5), 0, 1)

    # === Time Context Features (sin/cos encoding) ===
    # 21-22: Hour of day
    hours = np.array([datetime.fromtimestamp(ts / 1000).hour for ts in timestamps])
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    # 23-24: Day of week
    days = np.array([datetime.fromtimestamp(ts / 1000).weekday() for ts in timestamps])
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)

    # 25: Minute of hour (for intraday patterns)
    minutes = np.array([datetime.fromtimestamp(ts / 1000).minute for ts in timestamps])
    minute_sin = np.sin(2 * np.pi * minutes / 60)

    print("Features calculated!")

    return {
        # Price-based (need normalization)
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'returns': returns,
        'log_returns': log_returns,

        # Volatility (already normalized/scaled)
        'hl_spread': hl_spread,
        'vol_5': vol_5,
        'vol_20': vol_20,
        'atr': atr,

        # Volume
        'volume': volumes,
        'rel_volume': rel_volume,

        # Technical (already normalized)
        'sma_5_pct': sma_5_pct,
        'sma_20_pct': sma_20_pct,
        'sma_50_pct': sma_50_pct,
        'rsi': rsi,
        'macd_line': macd_line_norm,
        'macd_signal': macd_signal_norm,
        'macd_hist': macd_hist_norm,
        'bb_position': bb_position,

        # Time context (already -1 to 1)
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'minute_sin': minute_sin,
    }


def calculate_stats(features):
    """Calculate normalization statistics for price/volume"""
    return {
        'min_price': float(np.min([features['open'].min(), features['low'].min()])),
        'max_price': float(np.max([features['open'].max(), features['high'].max()])),
        'min_volume': float(features['volume'].min()),
        'max_volume': float(features['volume'].max()),
        'min_atr': float(np.nanmin(features['atr'])),
        'max_atr': float(np.nanmax(features['atr'])),
    }


def normalize_features(features, stats):
    """Normalize features to appropriate ranges"""
    n = len(features['close'])

    price_range = stats['max_price'] - stats['min_price'] or 1
    volume_range = stats['max_volume'] - stats['min_volume'] or 1
    atr_range = stats['max_atr'] - stats['min_atr'] or 1

    # Build feature matrix
    feature_matrix = np.zeros((n, CONFIG['features']), dtype=np.float32)

    # Price features (normalized 0-1)
    feature_matrix[:, 0] = (features['open'] - stats['min_price']) / price_range
    feature_matrix[:, 1] = (features['high'] - stats['min_price']) / price_range
    feature_matrix[:, 2] = (features['low'] - stats['min_price']) / price_range
    feature_matrix[:, 3] = (features['close'] - stats['min_price']) / price_range

    # Returns (already small, just clip)
    feature_matrix[:, 4] = np.clip(features['returns'] * 10, -1, 1)  # Scale up small returns
    feature_matrix[:, 5] = np.clip(features['log_returns'] * 10, -1, 1)

    # Volatility
    feature_matrix[:, 6] = np.clip(features['hl_spread'] * 20, 0, 1)
    feature_matrix[:, 7] = np.clip(features['vol_5'] * 100, 0, 1)
    feature_matrix[:, 8] = np.clip(features['vol_20'] * 100, 0, 1)
    feature_matrix[:, 9] = (features['atr'] - stats['min_atr']) / atr_range

    # Volume
    feature_matrix[:, 10] = (features['volume'] - stats['min_volume']) / volume_range
    feature_matrix[:, 11] = np.clip(features['rel_volume'] / 3, 0, 1)  # Cap at 3x avg

    # Technical indicators
    feature_matrix[:, 12] = np.clip(features['sma_5_pct'] * 10, -1, 1)
    feature_matrix[:, 13] = np.clip(features['sma_20_pct'] * 10, -1, 1)
    feature_matrix[:, 14] = np.clip(features['sma_50_pct'] * 10, -1, 1)
    feature_matrix[:, 15] = features['rsi']  # Already 0-1
    feature_matrix[:, 16] = np.clip(features['macd_line'] * 100, -1, 1)
    feature_matrix[:, 17] = np.clip(features['macd_signal'] * 100, -1, 1)
    feature_matrix[:, 18] = np.clip(features['macd_hist'] * 100, -1, 1)
    feature_matrix[:, 19] = features['bb_position']  # Already 0-1

    # Time features (already -1 to 1)
    feature_matrix[:, 20] = features['hour_sin']
    feature_matrix[:, 21] = features['hour_cos']
    feature_matrix[:, 22] = features['day_sin']
    feature_matrix[:, 23] = features['day_cos']
    feature_matrix[:, 24] = features['minute_sin']

    return feature_matrix


def create_sequences(feature_matrix, closes_normalized, sequence_length):
    """Create training sequences"""
    inputs = []
    outputs = []

    for i in range(sequence_length, len(feature_matrix)):
        inputs.append(feature_matrix[i - sequence_length:i])
        outputs.append(closes_normalized[i])

    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)


def build_model():
    """Build Enhanced LSTM model with attention"""
    model = tf.keras.Sequential([
        # First LSTM layer
        tf.keras.layers.LSTM(
            CONFIG['lstm_units'][0],
            return_sequences=True,
            input_shape=(CONFIG['sequence_length'], CONFIG['features'])
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Second LSTM layer
        tf.keras.layers.LSTM(CONFIG['lstm_units'][1], return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Third LSTM layer
        tf.keras.layers.LSTM(CONFIG['lstm_units'][2], return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(CONFIG['dropout_rate']),

        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
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
    print('S.U.P.I.D. Enhanced LSTM Model Training (25 Features)')
    print('=' * 60)
    print(f'Asset: {category}/{asset}')
    print(f'Epochs: {epochs}')
    print(f'Sequence Length: {CONFIG["sequence_length"]}')
    print(f'Features: {CONFIG["features"]}')
    print(f'Batch Size: {CONFIG["batch_size"]}')
    print('=' * 60)
    print()

    print('Features included:')
    print('  [1-4]   OHLC prices')
    print('  [5-6]   Returns & Log returns')
    print('  [7-10]  Volatility (H-L spread, Rolling std 5/20, ATR)')
    print('  [11-12] Volume (Raw, Relative)')
    print('  [13-15] Moving Averages (SMA 5/20/50)')
    print('  [16]    RSI')
    print('  [17-19] MACD (Line, Signal, Histogram)')
    print('  [20]    Bollinger Band position')
    print('  [21-25] Time encoding (Hour, Day, Minute)')
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

    # Engineer features
    print('Engineering features...')
    features = engineer_features(candles)

    # Calculate stats
    stats = calculate_stats(features)
    print(f'Price range: ${stats["min_price"]:.2f} - ${stats["max_price"]:.2f}')

    # Normalize
    print('Normalizing features...')
    feature_matrix = normalize_features(features, stats)
    closes_normalized = feature_matrix[:, 3]  # Close is at index 3

    # Create sequences
    print('Creating training sequences...')
    inputs, outputs = create_sequences(feature_matrix, closes_normalized, CONFIG['sequence_length'])
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
    print('Building enhanced model...')
    model = build_model()
    model.summary()

    print()
    print('=' * 60)
    print('Training started...')
    print('=' * 60)
    print()

    # Checkpoint directory for resume capability
    checkpoint_dir = MODELS_DIR / f'{asset.lower()}_enhanced_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'epoch_{epoch:03d}.weights.h5'

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
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
    model_name = f'{asset.lower()}_lstm_enhanced'
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
            'features': CONFIG['features'],
            'trained_at': datetime.now().isoformat(),
            'total_candles': len(candles),
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'feature_list': [
                'open', 'high', 'low', 'close', 'returns', 'log_returns',
                'hl_spread', 'vol_5', 'vol_20', 'atr',
                'volume', 'rel_volume',
                'sma_5_pct', 'sma_20_pct', 'sma_50_pct', 'rsi',
                'macd_line', 'macd_signal', 'macd_hist', 'bb_position',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin'
            ],
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
        print('Usage: python train_model_enhanced.py <category> <asset> [epochs]')
        print()
        print('Examples:')
        print('  python train_model_enhanced.py Crypto BTC 100')
        print('  python train_model_enhanced.py "Stock Market" AAPL 50')
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
