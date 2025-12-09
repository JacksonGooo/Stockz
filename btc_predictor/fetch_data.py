"""
Bitcoin Data Processor
Combines historical (Coinbase) + live (TradingView) data
Calculates 50+ technical indicators
Saves to organized daily/weekly files + master file

Usage:
  1. First run: py fetch_historical.py  (get 2 years of data from Coinbase)
  2. Start live: node live_collector.js  (collect ongoing data from TradingView)
  3. Process: py fetch_data.py  (combine all data and calculate features)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip
import os
import json
from pathlib import Path

def create_comprehensive_features(df):
    """Create 50+ technical indicators and features"""
    df = df.copy()

    print("  Calculating 50+ technical indicators...")

    # === PRICE METRICS ===
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['weighted_price'] = (df['high'] + df['low'] + 2*df['close']) / 4

    # === RETURNS ===
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_change'] = df['close'].diff()
    df['price_change_pct'] = df['price_change'] / df['close'].shift(1)
    df['rolling_returns_5'] = df['close'].pct_change(5)
    df['rolling_returns_15'] = df['close'].pct_change(15)
    df['rolling_returns_30'] = df['close'].pct_change(30)

    # === VOLATILITY ===
    df['realized_volatility'] = df['returns'].rolling(window=20).std()
    df['rolling_volatility_10'] = df['returns'].rolling(window=10).std()
    df['rolling_volatility_30'] = df['returns'].rolling(window=30).std()
    df['high_low_spread'] = df['high'] - df['low']
    df['high_low_spread_pct'] = df['high_low_spread'] / df['close']
    df['high_low_volatility'] = df['high_low_spread'].rolling(window=20).std()
    df['return_variance_20'] = df['returns'].rolling(window=20).var()

    # === PRICE RATIOS ===
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    df['open_close_ratio'] = df['open'] / df['close']

    # === MOVING AVERAGES ===
    for period in [7, 14, 25, 50, 99, 200]:
        df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'dist_from_ma_{period}'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']

    # === VOLUME ===
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    df['volume_volatility'] = df['volume'].rolling(window=20).std()
    df['vwap'] = (df['volume'] * df['typical_price']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['dist_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']

    # === MOMENTUM ===
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stochastic_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()

    # === BOLLINGER BANDS ===
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # === STATISTICAL ===
    df['skewness_20'] = df['returns'].rolling(window=20).skew()
    df['kurtosis_20'] = df['returns'].rolling(window=20).kurt()
    df['trend_strength'] = abs(df['close'] - df['ma_50']) / df['ma_50']

    return df


def get_week_start(date):
    return date - timedelta(days=date.weekday())


def load_asset_data(category, asset):
    """Load all data for a specific asset from unified Data/ structure"""
    all_candles = []

    asset_dir = Path(f'../Data/{category}/{asset}')
    if not asset_dir.exists():
        return None

    print(f"Loading {category}/{asset}...")
    for week_folder in sorted(asset_dir.glob('week_*')):
        for day_file in sorted(week_folder.glob('*.json')):
            with open(day_file, 'r') as f:
                day_data = json.load(f)
                all_candles.extend(day_data)

    if not all_candles:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    print(f"  Loaded: {len(df):,} candles")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")

    return df


def load_all_raw_data():
    """Load BTC data from unified Data/Crypto/BTC/ structure"""
    return load_asset_data('Crypto', 'BTC')


def process_and_save(df, category='Crypto', asset='BTC'):
    """Process data with features and save to unified Data/ structure"""
    print("\nCalculating comprehensive features...")
    df = create_comprehensive_features(df)
    print(f"Total features: {len(df.columns)}")

    # Save processed data to Data/Category/Asset/processed/
    processed_dir = Path(f'../Data/{category}/{asset}/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("\nOrganizing into daily/weekly files...")

    df['date'] = df.index.date
    dates = df['date'].unique()

    files_created = 0
    weeks_created = set()

    for date in dates:
        day_data = df[df['date'] == date].copy()
        day_data = day_data.drop(columns=['date'])

        week_start = get_week_start(pd.Timestamp(date))
        week_folder = processed_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
        week_folder.mkdir(exist_ok=True)
        weeks_created.add(week_folder)

        day_file = week_folder / f"{date}.pkl.gz"
        with gzip.open(day_file, 'wb') as f:
            pickle.dump(day_data, f)

        files_created += 1

    # Save master file in the asset directory
    df = df.drop(columns=['date'])
    master_file = Path(f'../Data/{category}/{asset}/{asset.lower()}_master.pkl.gz')
    with gzip.open(master_file, 'wb') as f:
        pickle.dump(df, f)

    master_size = os.path.getsize(master_file) / (1024 * 1024)

    print()
    print("="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Asset: {category}/{asset}")
    print(f"Total candles: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Daily files: {files_created}")
    print(f"Weekly folders: {len(weeks_created)}")
    print(f"Master file: {master_size:.2f} MB")
    print()
    print("="*60)
    print("Ready to train! Run: py train.py")
    print("="*60)

    return df


def load_master_data(category='Crypto', asset='BTC'):
    """Load the processed master file from unified structure"""
    master_file = Path(f'../Data/{category}/{asset}/{asset.lower()}_master.pkl.gz')
    if master_file.exists():
        with gzip.open(master_file, 'rb') as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    print("="*60)
    print("Bitcoin Data Processor")
    print("="*60)
    print()

    # Load all raw data
    df = load_all_raw_data()

    if df is None:
        print("No data found!")
        print()
        print("To get data:")
        print("  1. Historical: py fetch_historical.py")
        print("  2. Live: node live_collector.js")
        exit(1)

    # Process and save
    process_and_save(df)
