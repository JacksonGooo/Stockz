"""
Fast Historical Data Fetcher - Kraken
Fetches 2 years of BTC 1-minute data using Kraken API (720 candles/request)
Faster than Coinbase (300 candles/request)
"""

import ccxt
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

def get_week_start(date):
    """Get the Monday of the week for a given date"""
    return date - timedelta(days=date.weekday())

def fetch_day_data(exchange, date, base_dir):
    """Fetch one day of data and save to file"""
    day_str = date.strftime('%Y-%m-%d')
    week_start = get_week_start(date)
    week_folder = base_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
    day_file = week_folder / f"{day_str}.json"

    # Skip if already exists
    if day_file.exists():
        with open(day_file, 'r') as f:
            existing = json.load(f)
        return day_str, len(existing), True

    try:
        since = int(date.timestamp() * 1000)
        next_day = date + timedelta(days=1)
        until = int(next_day.timestamp() * 1000)

        day_candles = []
        fetch_since = since

        while fetch_since < until:
            candles = exchange.fetch_ohlcv(
                'BTC/USD',  # Kraken uses USD pairs
                timeframe='1m',
                since=fetch_since,
                limit=720  # Kraken allows 720
            )

            if not candles:
                break

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

            fetch_since = candles[-1][0] + 60000
            if candles[-1][0] >= until:
                break

            time.sleep(0.1)  # Rate limit delay

        if day_candles:
            week_folder.mkdir(parents=True, exist_ok=True)
            with open(day_file, 'w') as f:
                json.dump(day_candles, f)
            return day_str, len(day_candles), False

        return day_str, 0, False

    except Exception as e:
        return day_str, -1, str(e)

def fetch_historical_data(days=730):
    """
    Fetch historical BTC data from Kraken
    """
    print("="*60)
    print("FAST Historical Data Fetcher - Kraken")
    print("="*60)
    print(f"Fetching {days} days of BTC/USD 1-minute data")
    print("="*60)
    print()

    # Initialize Kraken (no API key needed for public data)
    print("[1/3] Connecting to Kraken...")
    exchange = ccxt.kraken({
        'rateLimit': 100,
        'enableRateLimit': True,
    })

    base_dir = Path('Data/Crypto/BTC')
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate list of dates to fetch
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [(end_date - timedelta(days=i)) for i in range(days)]
    dates.reverse()  # Oldest first

    print(f"\n[2/3] Fetching {len(dates)} days of data...")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print()

    total_candles = 0
    new_days = 0
    skipped_days = 0
    errors = 0
    start_time = time.time()

    # Process sequentially to respect rate limits
    for i, date in enumerate(dates):
        progress = ((i + 1) / len(dates)) * 100
        result = fetch_day_data(exchange, date, base_dir)

        day_str, count, was_cached = result

        if isinstance(was_cached, str):  # Error
            print(f"  [{progress:5.1f}%] {day_str}: Error - {was_cached[:30]}")
            errors += 1
        elif was_cached:
            skipped_days += 1
            total_candles += count
            if i % 50 == 0:  # Print every 50th skipped day
                print(f"  [{progress:5.1f}%] {day_str}: Cached ({count} candles)")
        else:
            new_days += 1
            total_candles += count
            print(f"  [{progress:5.1f}%] {day_str}: Downloaded {count} candles")

        # Progress update every 25 days
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(dates) - i - 1) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {i+1}/{len(dates)} days, "
                  f"~{remaining/60:.1f} min remaining ---\n")

    elapsed = time.time() - start_time

    print()
    print("="*60)
    print("HISTORICAL DATA COMPLETE!")
    print("="*60)
    print(f"Total candles: {total_candles:,}")
    print(f"New days downloaded: {new_days}")
    print(f"Days from cache: {skipped_days}")
    print(f"Errors: {errors}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Location: Data/Crypto/BTC/")
    print()
    print("="*60)
    print("Next step: py fetch_data.py  (to process with indicators)")
    print("="*60)

    return total_candles

if __name__ == "__main__":
    fetch_historical_data(days=730)  # 2 years
