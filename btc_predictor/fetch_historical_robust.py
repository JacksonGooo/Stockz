"""
Robust Historical Data Fetcher - Coinbase
Fetches BTC 1-minute data with proper retry logic and rate limiting
Fills gaps in existing data
"""

import ccxt
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import random

def get_week_start(date):
    """Get the Monday of the week for a given date"""
    return date - timedelta(days=date.weekday())

def fetch_day_with_retry(exchange, date, base_dir, max_retries=3):
    """Fetch one day of data with retry logic"""
    day_str = date.strftime('%Y-%m-%d')
    week_start = get_week_start(date)
    week_folder = base_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
    day_file = week_folder / f"{day_str}.json"

    # Skip if already exists and has data
    if day_file.exists():
        try:
            with open(day_file, 'r') as f:
                existing = json.load(f)
            if len(existing) > 100:  # Consider valid if > 100 candles
                return day_str, len(existing), "cached"
        except:
            pass  # Corrupt file, refetch

    since = int(date.timestamp() * 1000)
    next_day = date + timedelta(days=1)
    until = int(next_day.timestamp() * 1000)

    for attempt in range(max_retries):
        try:
            day_candles = []
            fetch_since = since

            while fetch_since < until:
                # Coinbase allows 300 candles per request
                candles = exchange.fetch_ohlcv(
                    'BTC/USD',
                    timeframe='1m',
                    since=fetch_since,
                    limit=300
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

                # Rate limiting - be nice to the API
                time.sleep(0.3 + random.uniform(0, 0.2))

            if day_candles:
                week_folder.mkdir(parents=True, exist_ok=True)
                with open(day_file, 'w') as f:
                    json.dump(day_candles, f)
                return day_str, len(day_candles), "downloaded"

            return day_str, 0, "no_data"

        except Exception as e:
            error_msg = str(e)
            if "Too many" in error_msg or "rate" in error_msg.lower():
                # Rate limited - exponential backoff
                wait_time = (2 ** attempt) * 5 + random.uniform(0, 5)
                print(f"    Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # Other error
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return day_str, -1, f"error: {error_msg[:50]}"

    return day_str, -1, "max_retries"

def fetch_missing_data(days=730):
    """
    Fetch missing BTC data from Coinbase
    Only fetches days that don't have valid data
    """
    print("="*60)
    print("Robust Historical Data Fetcher - Coinbase")
    print("="*60)
    print(f"Checking {days} days for missing data")
    print("="*60)
    print()

    # Initialize Coinbase
    print("[1/4] Connecting to Coinbase...")
    exchange = ccxt.coinbase({
        'rateLimit': 400,
        'enableRateLimit': True,
    })

    base_dir = Path('../Data/Crypto/BTC')
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate list of dates
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [(end_date - timedelta(days=i)) for i in range(days)]
    dates.reverse()

    print(f"\n[2/4] Scanning for missing days...")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Find missing days
    missing_dates = []
    cached_count = 0
    for date in dates:
        day_str = date.strftime('%Y-%m-%d')
        week_start = get_week_start(date)
        week_folder = base_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
        day_file = week_folder / f"{day_str}.json"

        if day_file.exists():
            try:
                with open(day_file, 'r') as f:
                    data = json.load(f)
                if len(data) > 100:
                    cached_count += 1
                    continue
            except:
                pass
        missing_dates.append(date)

    print(f"Found {cached_count} cached days")
    print(f"Found {len(missing_dates)} missing days to fetch")

    if not missing_dates:
        print("\nAll data already cached!")
        return 0

    print(f"\n[3/4] Fetching {len(missing_dates)} missing days...")
    print()

    total_candles = 0
    downloaded = 0
    errors = 0
    start_time = time.time()

    for i, date in enumerate(missing_dates):
        progress = ((i + 1) / len(missing_dates)) * 100
        result = fetch_day_with_retry(exchange, date, base_dir)

        day_str, count, status = result

        if status == "cached":
            pass  # Already counted
        elif status == "downloaded":
            downloaded += 1
            total_candles += count
            print(f"  [{progress:5.1f}%] {day_str}: {count} candles")
        elif status == "no_data":
            print(f"  [{progress:5.1f}%] {day_str}: No data available")
        else:
            errors += 1
            print(f"  [{progress:5.1f}%] {day_str}: {status}")

        # Progress update every 20 days
        if (i + 1) % 20 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(missing_dates) - i - 1) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {i+1}/{len(missing_dates)}, "
                  f"~{remaining/60:.1f} min remaining ---\n")

    elapsed = time.time() - start_time

    print()
    print("="*60)
    print("[4/4] FETCH COMPLETE!")
    print("="*60)
    print(f"Days downloaded: {downloaded}")
    print(f"Candles fetched: {total_candles:,}")
    print(f"Errors: {errors}")
    print(f"Previously cached: {cached_count}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Location: Data/Crypto/BTC/")
    print()
    print("="*60)
    print("Next: py fetch_data.py  (process with indicators)")
    print("="*60)

    return total_candles

if __name__ == "__main__":
    fetch_missing_data(days=730)
