"""
Multi-Asset Historical Data Fetcher
Fetches historical data for multiple assets:
- Crypto: BTC, ETH
- Stocks: SPY, AAPL (via TradingView)
- Commodities: GOLD
- Currencies: EURUSD
"""

import ccxt
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import random

# Asset configurations
ASSETS = {
    'Crypto': {
        'BTC': {'exchange': 'coinbase', 'symbol': 'BTC/USD'},
        'ETH': {'exchange': 'coinbase', 'symbol': 'ETH/USD'},
    },
    # Note: Stocks, Commodities, Currencies require TradingView (use universal_collector.js)
}

def get_week_start(date):
    return date - timedelta(days=date.weekday())

def fetch_day_with_retry(exchange, symbol, date, base_dir, category, asset, max_retries=3):
    """Fetch one day of data with retry logic"""
    day_str = date.strftime('%Y-%m-%d')
    week_start = get_week_start(date)
    week_folder = base_dir / f"week_{week_start.strftime('%Y-%m-%d')}"
    day_file = week_folder / f"{day_str}.json"

    if day_file.exists():
        try:
            with open(day_file, 'r') as f:
                existing = json.load(f)
            if len(existing) > 100:
                return day_str, len(existing), "cached"
        except:
            pass

    since = int(date.timestamp() * 1000)
    next_day = date + timedelta(days=1)
    until = int(next_day.timestamp() * 1000)

    for attempt in range(max_retries):
        try:
            day_candles = []
            fetch_since = since

            while fetch_since < until:
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
                wait_time = (2 ** attempt) * 5 + random.uniform(0, 5)
                print(f"    Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return day_str, -1, f"error: {error_msg[:50]}"

    return day_str, -1, "max_retries"

def fetch_asset_data(category, asset, config, days=730):
    """Fetch historical data for a single asset"""
    print(f"\n{'='*60}")
    print(f"Fetching {category}/{asset}")
    print(f"{'='*60}")

    # Initialize exchange
    if config['exchange'] == 'coinbase':
        exchange = ccxt.coinbase({
            'rateLimit': 400,
            'enableRateLimit': True,
        })
    else:
        print(f"Unknown exchange: {config['exchange']}")
        return 0

    base_dir = Path(f'Data/{category}/{asset}')
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate dates
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [(end_date - timedelta(days=i)) for i in range(days)]
    dates.reverse()

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

    print(f"Cached: {cached_count} days")
    print(f"Missing: {len(missing_dates)} days")

    if not missing_dates:
        print("All data cached!")
        return 0

    total_candles = 0
    downloaded = 0
    start_time = time.time()

    for i, date in enumerate(missing_dates):
        progress = ((i + 1) / len(missing_dates)) * 100
        result = fetch_day_with_retry(exchange, config['symbol'], date, base_dir, category, asset)
        day_str, count, status = result

        if status == "downloaded":
            downloaded += 1
            total_candles += count
            print(f"  [{progress:5.1f}%] {day_str}: {count} candles")
        elif status != "cached":
            print(f"  [{progress:5.1f}%] {day_str}: {status}")

        if (i + 1) % 20 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(missing_dates) - i - 1) / rate if rate > 0 else 0
            print(f"\n  --- {category}/{asset}: {i+1}/{len(missing_dates)}, ~{remaining/60:.1f} min left ---\n")

    print(f"\n{category}/{asset}: Downloaded {downloaded} days, {total_candles:,} candles")
    return total_candles

def fetch_all_assets(days=730):
    """Fetch data for all configured assets"""
    print("="*60)
    print("Multi-Asset Historical Data Fetcher")
    print("="*60)
    print(f"Days: {days}")
    print("="*60)

    total_candles = 0

    for category, assets in ASSETS.items():
        for asset, config in assets.items():
            candles = fetch_asset_data(category, asset, config, days)
            total_candles += candles

    print("\n" + "="*60)
    print("ALL ASSETS COMPLETE!")
    print("="*60)
    print(f"Total candles fetched: {total_candles:,}")
    print()
    print("For stocks/commodities/currencies, run:")
    print("  node universal_collector.js")
    print()
    print("To process with indicators:")
    print("  py fetch_data.py")
    print("="*60)

if __name__ == "__main__":
    fetch_all_assets(days=730)
