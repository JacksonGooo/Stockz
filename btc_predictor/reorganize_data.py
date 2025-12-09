"""
Reorganize Data Folder Structure

Moves all data into unified structure:
Data/
  Crypto/
    BTC/
      week_YYYY-MM-DD/
        YYYY-MM-DD.json
    ETH/
      ...
  Stocks/
    AAPL/
    SPY/
  Commodities/
    GOLD/
  Currencies/
    EURUSD/
"""

import os
import shutil
import json
from pathlib import Path

DATA_DIR = Path('Data')

def merge_json_files(src, dst):
    """Merge two JSON files containing candle arrays"""
    src_data = []
    dst_data = []

    if src.exists():
        with open(src, 'r') as f:
            src_data = json.load(f)

    if dst.exists():
        with open(dst, 'r') as f:
            dst_data = json.load(f)

    # Merge and deduplicate by timestamp
    all_data = dst_data + src_data
    seen = set()
    unique_data = []
    for candle in all_data:
        ts = candle.get('timestamp')
        if ts not in seen:
            seen.add(ts)
            unique_data.append(candle)

    # Sort by timestamp
    unique_data.sort(key=lambda x: x.get('timestamp', 0))

    # Write to destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(unique_data, f, indent=2)

    return len(unique_data)

def move_week_folder(src_week, dst_asset_dir):
    """Move a week folder to the asset directory, merging if needed"""
    if not src_week.exists():
        return

    week_name = src_week.name
    dst_week = dst_asset_dir / week_name

    for day_file in src_week.glob('*.json'):
        dst_file = dst_week / day_file.name
        count = merge_json_files(day_file, dst_file)
        print(f"  Merged {day_file.name} -> {count} candles")

def reorganize():
    print("="*60)
    print("Reorganizing Data Folder Structure")
    print("="*60)
    print()

    # Step 1: Move historical BTC data to Crypto/BTC
    historical_dir = DATA_DIR / 'historical'
    btc_dir = DATA_DIR / 'Crypto' / 'BTC'
    btc_dir.mkdir(parents=True, exist_ok=True)

    if historical_dir.exists():
        print("[1] Moving historical BTC data to Crypto/BTC...")
        for week_folder in historical_dir.glob('week_*'):
            move_week_folder(week_folder, btc_dir)
        print(f"  Done - moved {len(list(historical_dir.glob('week_*')))} week folders")

    # Step 2: Move live BTC data to Crypto/BTC
    live_dir = DATA_DIR / 'live'
    if live_dir.exists():
        print("\n[2] Merging live BTC data into Crypto/BTC...")
        for week_folder in live_dir.glob('week_*'):
            move_week_folder(week_folder, btc_dir)
        print("  Done")

    # Step 3: Check and move loose week folders at root
    print("\n[3] Checking loose week folders at root...")
    for week_folder in DATA_DIR.glob('week_*'):
        # Check what kind of data this contains
        for day_file in week_folder.glob('*.json'):
            try:
                with open(day_file, 'r') as f:
                    data = json.load(f)
                if data and isinstance(data, list) and len(data) > 0:
                    # Check if this looks like BTC data (high prices)
                    first_price = data[0].get('close', 0)
                    if first_price > 10000:  # BTC range
                        print(f"  {week_folder.name}: BTC data (price ~${first_price:.0f})")
                        move_week_folder(week_folder, btc_dir)
                    else:
                        print(f"  {week_folder.name}: Unknown (price ~${first_price:.2f})")
            except Exception as e:
                print(f"  {week_folder.name}: Error - {e}")

    # Step 4: Rename "Stock Market" to "Stocks"
    stock_market_dir = DATA_DIR / 'Stock Market'
    stocks_dir = DATA_DIR / 'Stocks'
    if stock_market_dir.exists():
        print("\n[4] Renaming 'Stock Market' to 'Stocks'...")
        if stocks_dir.exists():
            # Merge contents
            for asset_folder in stock_market_dir.iterdir():
                if asset_folder.is_dir():
                    dst_asset = stocks_dir / asset_folder.name
                    dst_asset.mkdir(parents=True, exist_ok=True)
                    for week_folder in asset_folder.glob('week_*'):
                        move_week_folder(week_folder, dst_asset)
        else:
            shutil.move(str(stock_market_dir), str(stocks_dir))
        print("  Done")

    # Step 5: Clean up old folders and files
    print("\n[5] Cleaning up old folders and files...")

    to_remove = [
        DATA_DIR / 'historical',
        DATA_DIR / 'live',
        DATA_DIR / 'Stock Market',
        DATA_DIR / 'btc_master.pkl.gz',
        DATA_DIR / 'btc_raw_data.json',
    ]

    # Also remove loose week folders
    for week_folder in DATA_DIR.glob('week_*'):
        to_remove.append(week_folder)

    for item in to_remove:
        if item.exists():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"  Removed: {item.name}")
            except Exception as e:
                print(f"  Failed to remove {item.name}: {e}")

    # Step 6: Count final data
    print("\n" + "="*60)
    print("REORGANIZATION COMPLETE!")
    print("="*60)
    print("\nNew structure:")

    for category in ['Crypto', 'Stocks', 'Commodities', 'Currencies']:
        cat_dir = DATA_DIR / category
        if cat_dir.exists():
            assets = [d.name for d in cat_dir.iterdir() if d.is_dir()]
            print(f"\n{category}/")
            for asset in assets:
                asset_dir = cat_dir / asset
                weeks = list(asset_dir.glob('week_*'))
                total_candles = 0
                for week in weeks:
                    for day_file in week.glob('*.json'):
                        try:
                            with open(day_file, 'r') as f:
                                data = json.load(f)
                            total_candles += len(data)
                        except:
                            pass
                print(f"  {asset}/: {len(weeks)} weeks, {total_candles:,} candles")

    print()

if __name__ == "__main__":
    reorganize()
