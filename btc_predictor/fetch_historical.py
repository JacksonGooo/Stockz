"""
Historical Data Fetcher - Coinbase
Fetches 2 years of BTC 1-minute data from Coinbase
Organizes by: daily files inside weekly folders + master file
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip
import os
import time
from pathlib import Path

def get_week_start(date):
    """Get the Monday of the week for a given date"""
    return date - timedelta(days=date.weekday())

def fetch_historical_data(days=730):
    """
    Fetch 2 years of Bitcoin historical data from Coinbase
    1 day at a time, 1440 candles per day
    """
    print("="*60)
    print("Historical Data Fetcher - Coinbase")
    print("="*60)
    print(f"Fetching {days} days of BTC/USD 1-minute data")
    print("="*60)
    print()

    # Initialize Coinbase
    print("[1/3] Connecting to Coinbase...")
    exchange = ccxt.coinbase({
        'rateLimit': 400,
        'enableRateLimit': True,
    })

    # Unified structure: Data/Crypto/BTC/
    base_dir = Path('Data/Crypto/BTC')
    base_dir.mkdir(parents=True, exist_ok=True)

    all_candles = []
    files_created = 0
    weeks_created = set()

    print(f"\n[2/3] Fetching {days} days of data...\n")

    # Start from 2 years ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current_date = start_date

    while current_date < end_date:
        progress = ((current_date - start_date).days / days) * 100
        day_str = current_date.strftime('%Y-%m-%d')

        # Check if we already have this day
        week_start = get_week_start(current_date)
        week_folder = base_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
        day_file = week_folder / f"{day_str}.json"

        if day_file.exists():
            print(f"  [{progress:5.1f}%] {day_str}: Already exists, skipping", flush=True)
            # Load existing for master file
            import json
            with open(day_file, 'r') as f:
                day_candles = json.load(f)
                all_candles.extend(day_candles)
            files_created += 1
            weeks_created.add(week_folder)
            current_date += timedelta(days=1)
            continue

        print(f"  [{progress:5.1f}%] {day_str}: Fetching...", end='', flush=True)

        try:
            # Fetch data for this day
            since = int(current_date.timestamp() * 1000)
            next_day = current_date + timedelta(days=1)
            until = int(next_day.timestamp() * 1000)

            day_candles = []
            fetch_since = since

            # Coinbase allows 300 candles per request
            while fetch_since < until:
                candles = exchange.fetch_ohlcv(
                    'BTC/USD',
                    timeframe='1m',
                    since=fetch_since,
                    limit=300
                )

                if not candles:
                    break

                # Filter to only this day
                for c in candles:
                    if c[0] < until:
                        day_candles.append({
                            'timestamp': c[0],
                            'open': c[1],
                            'high': c[2],
                            'low': c[3],
                            'close': c[4],
                            'volume': c[5]
                        })

                fetch_since = candles[-1][0] + 60000  # Next minute

                if candles[-1][0] >= until:
                    break

                time.sleep(0.1)  # Rate limiting

            if len(day_candles) > 0:
                # Create week folder
                week_folder.mkdir(exist_ok=True)
                weeks_created.add(week_folder)

                # Save day file
                import json
                with open(day_file, 'w') as f:
                    json.dump(day_candles, f)

                all_candles.extend(day_candles)
                files_created += 1
                print(f" {len(day_candles)} candles [OK]", flush=True)
            else:
                print(f" No data", flush=True)

        except Exception as e:
            print(f" Error: {str(e)[:40]}", flush=True)

        current_date += timedelta(days=1)
        time.sleep(0.2)  # Be nice to the API

    # No separate master file - all data is in daily files
    print(f"\n[3/3] All data saved to daily files...")

    # Summary
    print()
    print("="*60)
    print("HISTORICAL DATA COMPLETE!")
    print("="*60)
    print(f"Total candles: {len(all_candles):,}")
    print(f"Daily files: {files_created}")
    print(f"Weekly folders: {len(weeks_created)}")
    print(f"Location: Data/Crypto/BTC/")
    if all_candles:
        print(f"Date range: {datetime.fromtimestamp(all_candles[0]['timestamp']/1000)} to {datetime.fromtimestamp(all_candles[-1]['timestamp']/1000)}")
    print()
    print("="*60)
    print("Now run: py fetch_data.py  (to process features)")
    print("="*60)

    return all_candles


if __name__ == "__main__":
    fetch_historical_data(days=730)  # 2 years
