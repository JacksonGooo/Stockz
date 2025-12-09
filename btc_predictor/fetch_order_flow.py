#!/usr/bin/env python3
"""
Binance Order Flow Data Fetcher
Fetches aggregate trades to calculate buy/sell pressure and order flow metrics

Order flow features we'll calculate:
- Taker buy vs taker sell volume ratio
- Large trade detection (whale activity)
- Trade count per minute
- Average trade size
- Buy/sell imbalance (cumulative delta)
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Binance API endpoints
BINANCE_BASE = "https://api.binance.com"
AGG_TRADES_URL = f"{BINANCE_BASE}/api/v3/aggTrades"

# Data directory
DATA_DIR = Path(__file__).parent.parent / 'Data' / 'Crypto' / 'BTC' / 'order_flow'

# Rate limiting
REQUEST_DELAY = 0.1  # 100ms between requests


def fetch_agg_trades(symbol='BTCUSDT', start_time=None, end_time=None, limit=1000):
    """
    Fetch aggregate trades from Binance

    Returns list of trades:
    - a: Aggregate tradeId
    - p: Price
    - q: Quantity
    - f: First tradeId
    - l: Last tradeId
    - T: Timestamp
    - m: Was the buyer the maker? (True = sell, False = buy)
    - M: Was the trade the best price match?
    """
    params = {
        'symbol': symbol,
        'limit': limit
    }

    if start_time:
        params['startTime'] = int(start_time)
    if end_time:
        params['endTime'] = int(end_time)

    try:
        response = requests.get(AGG_TRADES_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching trades: {e}")
        return []


def calculate_minute_order_flow(trades):
    """
    Calculate order flow metrics for a minute of trades

    m=True means the buyer was the maker (so this is a SELL/taker sell)
    m=False means the seller was the maker (so this is a BUY/taker buy)
    """
    if not trades:
        return None

    buy_volume = 0
    sell_volume = 0
    buy_count = 0
    sell_count = 0
    buy_sizes = []
    sell_sizes = []

    for trade in trades:
        qty = float(trade['q'])
        price = float(trade['p'])
        value = qty * price

        if trade['m']:  # Buyer is maker = SELL
            sell_volume += value
            sell_count += 1
            sell_sizes.append(qty)
        else:  # Seller is maker = BUY
            buy_volume += value
            buy_count += 1
            buy_sizes.append(qty)

    total_volume = buy_volume + sell_volume
    total_count = buy_count + sell_count

    return {
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'total_volume': total_volume,
        'buy_sell_ratio': buy_volume / (sell_volume + 1e-10),
        'volume_imbalance': (buy_volume - sell_volume) / (total_volume + 1e-10),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'total_count': total_count,
        'avg_buy_size': sum(buy_sizes) / len(buy_sizes) if buy_sizes else 0,
        'avg_sell_size': sum(sell_sizes) / len(sell_sizes) if sell_sizes else 0,
        'max_buy_size': max(buy_sizes) if buy_sizes else 0,
        'max_sell_size': max(sell_sizes) if sell_sizes else 0,
        'large_buys': sum(1 for s in buy_sizes if s > 0.1),  # > 0.1 BTC
        'large_sells': sum(1 for s in sell_sizes if s > 0.1),
    }


def fetch_order_flow_for_day(symbol='BTCUSDT', date=None):
    """
    Fetch order flow data for an entire day, minute by minute
    """
    if date is None:
        date = datetime.now().date() - timedelta(days=1)

    start_dt = datetime.combine(date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching order flow for {date}...")

    order_flow_data = []
    current_ms = start_ms

    while current_ms < end_ms:
        minute_end = current_ms + 60000  # 1 minute in ms

        # Fetch trades for this minute
        trades = fetch_agg_trades(
            symbol=symbol,
            start_time=current_ms,
            end_time=minute_end - 1,
            limit=1000
        )

        if trades:
            metrics = calculate_minute_order_flow(trades)
            if metrics:
                metrics['timestamp'] = current_ms
                order_flow_data.append(metrics)

        current_ms = minute_end
        time.sleep(REQUEST_DELAY)

        # Progress indicator
        if len(order_flow_data) % 60 == 0:
            progress = (current_ms - start_ms) / (end_ms - start_ms) * 100
            print(f"  Progress: {progress:.1f}% ({len(order_flow_data)} minutes)")

    return order_flow_data


def save_order_flow(data, date):
    """Save order flow data to JSON file"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = DATA_DIR / f"{date.strftime('%Y-%m-%d')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} minutes of order flow to {filename}")
    return filename


def fetch_recent_order_flow(symbol='BTCUSDT', minutes=60):
    """
    Fetch order flow for the most recent N minutes
    Useful for live predictions
    """
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = end_ms - (minutes * 60 * 1000)

    print(f"Fetching last {minutes} minutes of order flow...")

    order_flow_data = []
    current_ms = start_ms

    while current_ms < end_ms:
        minute_end = current_ms + 60000

        trades = fetch_agg_trades(
            symbol=symbol,
            start_time=current_ms,
            end_time=minute_end - 1,
            limit=1000
        )

        if trades:
            metrics = calculate_minute_order_flow(trades)
            if metrics:
                metrics['timestamp'] = current_ms
                order_flow_data.append(metrics)

        current_ms = minute_end
        time.sleep(REQUEST_DELAY)

    print(f"Fetched {len(order_flow_data)} minutes of order flow data")
    return order_flow_data


def main():
    print("=" * 60)
    print("BINANCE ORDER FLOW DATA FETCHER")
    print("=" * 60)
    print()

    # Test with recent data
    print("Testing with last 10 minutes...")
    recent = fetch_recent_order_flow(minutes=10)

    if recent:
        print("\nSample order flow data:")
        for metric in recent[-3:]:
            timestamp = datetime.fromtimestamp(metric['timestamp'] / 1000)
            print(f"\n{timestamp}:")
            print(f"  Buy/Sell Volume: ${metric['buy_volume']:,.0f} / ${metric['sell_volume']:,.0f}")
            print(f"  Buy/Sell Ratio: {metric['buy_sell_ratio']:.2f}")
            print(f"  Volume Imbalance: {metric['volume_imbalance']:.3f}")
            print(f"  Trade Count: {metric['total_count']} (Buy: {metric['buy_count']}, Sell: {metric['sell_count']})")
            print(f"  Large Trades: {metric['large_buys']} buys, {metric['large_sells']} sells")

    print("\n" + "=" * 60)
    print("To fetch historical data, use:")
    print("  fetch_order_flow_for_day(date=datetime(2024, 1, 15).date())")
    print("=" * 60)


if __name__ == '__main__':
    main()
