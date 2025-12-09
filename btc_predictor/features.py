"""
Feature Engineering for BTC Price Prediction
Calculates 20 features from OHLCV data for the autoregressive LSTM model
"""

import numpy as np
import pandas as pd


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal line, and Histogram"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands and width"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = (upper - lower) / sma  # Normalized width
    return upper, lower, width


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_obv_change(close, volume):
    """Calculate On-Balance Volume change (normalized)"""
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_change = obv.pct_change()
    return obv_change


def calculate_vwap_deviation(high, low, close, volume):
    """Calculate VWAP deviation from current price"""
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    vwap = cumulative_tp_vol / cumulative_vol
    deviation = (close - vwap) / vwap
    return deviation


def encode_time_features(timestamps):
    """Encode temporal features cyclically"""
    # Convert to datetime if needed
    dt = pd.to_datetime(timestamps, unit='ms')

    # Use .dt accessor for Series
    hour = dt.dt.hour + dt.dt.minute / 60
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Minute of day (normalized 0-1)
    minute_of_day = (dt.dt.hour * 60 + dt.dt.minute) / (24 * 60)

    return hour_sin, hour_cos, minute_of_day


def calculate_all_features(df):
    """
    Calculate all 20 features from OHLCV DataFrame

    Input DataFrame must have columns: timestamp, open, high, low, close, volume
    Returns DataFrame with all 20 features
    """
    features = pd.DataFrame(index=df.index)

    # 1-5: OHLCV as returns (percentage change from previous close)
    prev_close = df['close'].shift(1)
    features['open_return'] = (df['open'] - prev_close) / prev_close
    features['high_return'] = (df['high'] - prev_close) / prev_close
    features['low_return'] = (df['low'] - prev_close) / prev_close
    features['close_return'] = (df['close'] - prev_close) / prev_close
    features['volume_change'] = df['volume'].pct_change()

    # 6: RSI
    features['rsi'] = calculate_rsi(df['close']) / 100  # Normalize to 0-1

    # 7-9: MACD
    macd, macd_signal, macd_hist = calculate_macd(df['close'])
    # Normalize MACD by price
    features['macd'] = macd / df['close']
    features['macd_signal'] = macd_signal / df['close']
    features['macd_histogram'] = macd_hist / df['close']

    # 10-11: EMAs (as deviation from price)
    ema_9 = df['close'].ewm(span=9, adjust=False).mean()
    ema_21 = df['close'].ewm(span=21, adjust=False).mean()
    features['ema_9_dev'] = (df['close'] - ema_9) / ema_9
    features['ema_21_dev'] = (df['close'] - ema_21) / ema_21

    # 12-14: Bollinger Bands
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(df['close'])
    features['bb_upper_dev'] = (df['close'] - bb_upper) / df['close']
    features['bb_lower_dev'] = (df['close'] - bb_lower) / df['close']
    features['bb_width'] = bb_width

    # 15: ATR (normalized by price)
    atr = calculate_atr(df['high'], df['low'], df['close'])
    features['atr'] = atr / df['close']

    # 16: OBV change
    features['obv_change'] = calculate_obv_change(df['close'], df['volume'])

    # 17: VWAP deviation
    features['vwap_deviation'] = calculate_vwap_deviation(
        df['high'], df['low'], df['close'], df['volume']
    )

    # 18-20: Temporal features
    hour_sin, hour_cos, minute_of_day = encode_time_features(df['timestamp'])
    features['hour_sin'] = hour_sin
    features['hour_cos'] = hour_cos
    features['minute_of_day'] = minute_of_day

    # Replace infinities and clip extreme values
    features = features.replace([np.inf, -np.inf], np.nan)

    # Clip extreme outliers (beyond 5 std)
    for col in features.columns:
        if col not in ['hour_sin', 'hour_cos', 'minute_of_day', 'rsi']:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = features[col].clip(mean - 5*std, mean + 5*std)

    return features


def rolling_zscore(df, window=60):
    """
    Apply rolling z-score normalization to avoid lookahead bias
    """
    normalized = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col in ['hour_sin', 'hour_cos', 'minute_of_day']:
            # Don't normalize cyclical features
            normalized[col] = df[col]
        elif col == 'rsi':
            # RSI is already 0-1, just center it
            normalized[col] = df[col] - 0.5
        else:
            # Rolling z-score
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            normalized[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    return normalized


# Feature names for reference
FEATURE_NAMES = [
    'open_return', 'high_return', 'low_return', 'close_return', 'volume_change',
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'ema_9_dev', 'ema_21_dev',
    'bb_upper_dev', 'bb_lower_dev', 'bb_width',
    'atr', 'obv_change', 'vwap_deviation',
    'hour_sin', 'hour_cos', 'minute_of_day'
]

NUM_FEATURES = len(FEATURE_NAMES)


if __name__ == '__main__':
    # Test with sample data
    print(f"Feature engineering module loaded")
    print(f"Number of features: {NUM_FEATURES}")
    print(f"Features: {FEATURE_NAMES}")
