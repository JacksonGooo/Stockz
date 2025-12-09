"""
Historical Data Fetcher - All Crypto via Coinbase
Fetches 1-minute data for all crypto assets with proper retry logic
"""

import ccxt
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import random
import sys

# Crypto assets to fetch (Coinbase symbols)
CRYPTO_ASSETS = {
    'ETH': 'ETH/USD',
    'SOL': 'SOL/USD',
    'XRP': 'XRP/USD',
    'DOGE': 'DOGE/USD',
    'ADA': 'ADA/USD',
    'AVAX': 'AVAX/USD',
    'DOT': 'DOT/USD',
    'POL': 'POL/USD',  # Polygon
    'LINK': 'LINK/USD',
    'LTC': 'LTC/USD',
    'UNI': 'UNI/USD',
    'ATOM': 'ATOM/USD',
    'XLM': 'XLM/USD',
    'ALGO': 'ALGO/USD',
}

DATA_DIR = Path(__file__).parent.parent / 'Data' / 'Crypto'

def get_week_start(date):
    """Get the Monday of the week for a given date"""
    return date - timedelta(days=date.weekday())

def fetch_day_with_retry(exchange, symbol, asset_name, date, base_dir, max_retries=3):
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
                    symbol,
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

def fetch_crypto_historical(asset_name, symbol, days=365):
    """
    Fetch historical data for a crypto asset
    """
    base_dir = DATA_DIR / asset_name

    print(f"\n{'='*60}")
    print(f"Fetching {asset_name} ({symbol})")
    print(f"{'='*60}")

    # Initialize exchange
    print(f"[1/3] Connecting to Coinbase...")
    exchange = ccxt.coinbase({
        'enableRateLimit': True,
        'rateLimit': 400,
    })

    # Scan for missing days
    print(f"[2/3] Scanning for missing days...")
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    dates_to_check = []
    current = start_date
    while current < end_date:
        dates_to_check.append(current)
        current += timedelta(days=1)

    # Count cached
    cached = 0
    missing = []
    for date in dates_to_check:
        day_str = date.strftime('%Y-%m-%d')
        week_start = get_week_start(date)
        week_folder = base_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
        day_file = week_folder / f"{day_str}.json"

        if day_file.exists():
            try:
                with open(day_file, 'r') as f:
                    data = json.load(f)
                if len(data) > 100:
                    cached += 1
                    continue
            except:
                pass
        missing.append(date)

    print(f"  Found {cached} cached days")
    print(f"  Found {len(missing)} missing days to fetch")

    if not missing:
        print(f"  {asset_name} is up to date!")
        return {'cached': cached, 'fetched': 0, 'errors': 0}

    # Fetch missing days
    print(f"\n[3/3] Fetching {len(missing)} missing days...")
    fetched = 0
    errors = 0
    total_candles = 0
    start_time = time.time()

    for i, date in enumerate(missing):
        day_str, count, status = fetch_day_with_retry(exchange, symbol, asset_name, date, base_dir)

        if status == "downloaded":
            fetched += 1
            total_candles += count
            pct = ((i + 1) / len(missing)) * 100
            print(f"  [{pct:5.1f}%] {day_str}: {count} candles")
        elif status == "cached":
            pass
        elif "error" in status:
            errors += 1
            print(f"  [{((i + 1) / len(missing)) * 100:5.1f}%] {day_str}: {status}")

        # Progress update every 20 days
        if (i + 1) % 20 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(missing) - i - 1) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {i+1}/{len(missing)}, ~{remaining/60:.1f} min remaining ---\n")

    print(f"\n{asset_name} Complete: {fetched} days, {total_candles:,} candles, {errors} errors")
    return {'cached': cached, 'fetched': fetched, 'errors': errors, 'candles': total_candles}

def main():
    """Fetch all crypto assets"""
    print("="*60)
    print("Historical Data Fetcher - All Crypto via Coinbase")
    print("="*60)

    # Check for specific asset argument
    if len(sys.argv) > 1:
        asset = sys.argv[1].upper()
        if asset in CRYPTO_ASSETS:
            fetch_crypto_historical(asset, CRYPTO_ASSETS[asset], days=365)
            return
        else:
            print(f"Unknown asset: {asset}")
            print(f"Available: {', '.join(CRYPTO_ASSETS.keys())}")
            return

    # Fetch all assets
    results = {}
    for asset_name, symbol in CRYPTO_ASSETS.items():
        try:
            result = fetch_crypto_historical(asset_name, symbol, days=365)
            results[asset_name] = result
        except Exception as e:
            print(f"\nError fetching {asset_name}: {e}")
            results[asset_name] = {'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_fetched = 0
    total_candles = 0
    for asset, result in results.items():
        if 'error' in result:
            print(f"  {asset}: ERROR - {result['error'][:40]}")
        else:
            total_fetched += result.get('fetched', 0)
            total_candles += result.get('candles', 0)
            print(f"  {asset}: {result.get('cached', 0)} cached, {result.get('fetched', 0)} fetched")

    print(f"\nTotal: {total_fetched} days fetched, {total_candles:,} candles")

if __name__ == "__main__":
    main()
