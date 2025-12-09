"""
Data Gap Checker and Backfill Script
Detects gaps in collected data and fills them automatically.
Run this on startup to ensure continuous data coverage.

Usage:
    python backfill_gaps.py [--check-only]
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / 'Data'

# Coinbase API for crypto backfill
COINBASE_API = 'https://api.exchange.coinbase.com'

# Assets that can be backfilled via Coinbase
COINBASE_ASSETS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD',
    'XRP': 'XRP-USD',
    'DOGE': 'DOGE-USD',
    'ADA': 'ADA-USD',
    'AVAX': 'AVAX-USD',
    'DOT': 'DOT-USD',
    'MATIC': 'MATIC-USD',
    'LINK': 'LINK-USD',
    'UNI': 'UNI-USD',
    'ATOM': 'ATOM-USD',
    'LTC': 'LTC-USD',
    'XLM': 'XLM-USD',
}


def get_week_number(timestamp_ms):
    """Get week number from timestamp"""
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.isocalendar()[1]


def load_all_candles(asset_dir):
    """Load all candles for an asset and return sorted by timestamp"""
    all_candles = []

    if not asset_dir.exists():
        return []

    for week_dir in asset_dir.iterdir():
        if not week_dir.name.startswith('week_'):
            continue
        for json_file in week_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_candles.extend(data)
            except (json.JSONDecodeError, IOError):
                continue

    # Dedupe and sort
    unique = {c['timestamp']: c for c in all_candles}
    return sorted(unique.values(), key=lambda x: x['timestamp'])


def find_gaps(candles, max_gap_minutes=5):
    """Find gaps in candle data larger than max_gap_minutes"""
    gaps = []

    if len(candles) < 2:
        return gaps

    for i in range(1, len(candles)):
        prev_ts = candles[i-1]['timestamp']
        curr_ts = candles[i]['timestamp']
        gap_minutes = (curr_ts - prev_ts) / 60000

        if gap_minutes > max_gap_minutes:
            gaps.append({
                'start': prev_ts,
                'end': curr_ts,
                'minutes': int(gap_minutes),
                'start_dt': datetime.fromtimestamp(prev_ts / 1000).isoformat(),
                'end_dt': datetime.fromtimestamp(curr_ts / 1000).isoformat(),
            })

    return gaps


def fetch_coinbase_candles(product_id, start_ts, end_ts):
    """Fetch candles from Coinbase to fill a gap"""
    candles = []

    # Coinbase returns max 300 candles per request
    # Work backwards from end to start
    current_end = end_ts

    while current_end > start_ts:
        # Calculate start for this batch (max 300 minutes back)
        batch_start = max(start_ts, current_end - (300 * 60 * 1000))

        url = f'{COINBASE_API}/products/{product_id}/candles'
        params = {
            'start': datetime.fromtimestamp(batch_start / 1000).isoformat(),
            'end': datetime.fromtimestamp(current_end / 1000).isoformat(),
            'granularity': 60,  # 1 minute
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for row in data:
                    # Coinbase format: [timestamp, low, high, open, close, volume]
                    candles.append({
                        'timestamp': row[0] * 1000,  # Convert to ms
                        'open': float(row[3]),
                        'high': float(row[2]),
                        'low': float(row[1]),
                        'close': float(row[4]),
                        'volume': float(row[5]),
                    })
            else:
                print(f'  Warning: Coinbase returned {response.status_code}')
                break
        except Exception as e:
            print(f'  Error fetching from Coinbase: {e}')
            break

        current_end = batch_start
        time.sleep(0.1)  # Rate limiting

    return candles


def save_candles(asset_dir, candles):
    """Save candles to appropriate week folders"""
    if not candles:
        return 0

    # Group by week
    by_week = defaultdict(list)
    for candle in candles:
        week = get_week_number(candle['timestamp'])
        by_week[week].append(candle)

    saved = 0
    for week, week_candles in by_week.items():
        week_dir = asset_dir / f'week_{week:02d}'
        week_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp range for filename
        min_ts = min(c['timestamp'] for c in week_candles)
        max_ts = max(c['timestamp'] for c in week_candles)
        filename = f'backfill_{min_ts}_{max_ts}.json'

        filepath = week_dir / filename
        with open(filepath, 'w') as f:
            json.dump(sorted(week_candles, key=lambda x: x['timestamp']), f)

        saved += len(week_candles)

    return saved


def check_and_backfill_asset(category, asset, check_only=False):
    """Check for gaps and optionally backfill them"""
    asset_dir = DATA_DIR / category / asset
    candles = load_all_candles(asset_dir)

    if not candles:
        print(f'  No data found')
        return {'gaps': 0, 'filled': 0}

    # Get date range
    first_dt = datetime.fromtimestamp(candles[0]['timestamp'] / 1000)
    last_dt = datetime.fromtimestamp(candles[-1]['timestamp'] / 1000)

    print(f'  Data range: {first_dt.strftime("%Y-%m-%d %H:%M")} to {last_dt.strftime("%Y-%m-%d %H:%M")}')
    print(f'  Total candles: {len(candles):,}')

    # Check for gap between last candle and now
    now_ts = int(datetime.now().timestamp() * 1000)
    gap_to_now = (now_ts - candles[-1]['timestamp']) / 60000

    if gap_to_now > 5:
        print(f'  Gap to current time: {int(gap_to_now)} minutes')

    # Find internal gaps
    gaps = find_gaps(candles)

    if gaps:
        print(f'  Found {len(gaps)} gap(s):')
        for gap in gaps[:5]:  # Show first 5
            print(f'    - {gap["minutes"]} min gap: {gap["start_dt"]} to {gap["end_dt"]}')
        if len(gaps) > 5:
            print(f'    ... and {len(gaps) - 5} more')
    else:
        print(f'  No gaps found (data is continuous)')

    # Backfill if requested and asset is supported
    filled = 0
    if not check_only and gaps:
        coinbase_symbol = COINBASE_ASSETS.get(asset)
        if coinbase_symbol:
            print(f'  Backfilling from Coinbase...')
            for gap in gaps:
                new_candles = fetch_coinbase_candles(
                    coinbase_symbol,
                    gap['start'],
                    gap['end']
                )
                if new_candles:
                    saved = save_candles(asset_dir, new_candles)
                    filled += saved
                    print(f'    Filled {saved} candles for gap at {gap["start_dt"]}')
        else:
            print(f'  Cannot backfill (no API source for {asset})')

    # Also try to fill gap to current time for crypto
    if not check_only and gap_to_now > 5:
        coinbase_symbol = COINBASE_ASSETS.get(asset)
        if coinbase_symbol:
            print(f'  Backfilling gap to current time...')
            new_candles = fetch_coinbase_candles(
                coinbase_symbol,
                candles[-1]['timestamp'],
                now_ts
            )
            if new_candles:
                saved = save_candles(asset_dir, new_candles)
                filled += saved
                print(f'    Filled {saved} candles up to now')

    return {'gaps': len(gaps), 'filled': filled}


def main():
    check_only = '--check-only' in sys.argv

    print('=' * 60)
    print('S.U.P.I.D. Data Gap Checker')
    print('=' * 60)
    print(f'Mode: {"Check Only" if check_only else "Check and Backfill"}')
    print(f'Data directory: {DATA_DIR}')
    print()

    total_gaps = 0
    total_filled = 0

    # Scan all categories
    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        print(f'\n[{category}]')

        for asset_dir in category_dir.iterdir():
            if not asset_dir.is_dir():
                continue

            asset = asset_dir.name
            print(f'\n{asset}:')

            result = check_and_backfill_asset(category, asset, check_only)
            total_gaps += result['gaps']
            total_filled += result['filled']

    print()
    print('=' * 60)
    print(f'Summary: Found {total_gaps} gaps across all assets')
    if not check_only:
        print(f'Filled: {total_filled} candles')
    print('=' * 60)


if __name__ == '__main__':
    main()
