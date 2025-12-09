"""
Asset Configuration
Define all assets to track across categories
"""

# All assets organized by category
ASSETS = {
    "Crypto": {
        "BTC": {"symbol": "COINBASE:BTCUSD", "exchange": "coinbase", "pair": "BTC/USD"},
        "ETH": {"symbol": "COINBASE:ETHUSD", "exchange": "coinbase", "pair": "ETH/USD"},
        "SOL": {"symbol": "COINBASE:SOLUSD", "exchange": "coinbase", "pair": "SOL/USD"},
        "XRP": {"symbol": "COINBASE:XRPUSD", "exchange": "coinbase", "pair": "XRP/USD"},
        "DOGE": {"symbol": "COINBASE:DOGEUSD", "exchange": "coinbase", "pair": "DOGE/USD"},
    },
    "Stocks": {
        "AAPL": {"symbol": "NASDAQ:AAPL", "exchange": "nasdaq", "pair": "AAPL"},
        "GOOGL": {"symbol": "NASDAQ:GOOGL", "exchange": "nasdaq", "pair": "GOOGL"},
        "MSFT": {"symbol": "NASDAQ:MSFT", "exchange": "nasdaq", "pair": "MSFT"},
        "AMZN": {"symbol": "NASDAQ:AMZN", "exchange": "nasdaq", "pair": "AMZN"},
        "TSLA": {"symbol": "NASDAQ:TSLA", "exchange": "nasdaq", "pair": "TSLA"},
        "NVDA": {"symbol": "NASDAQ:NVDA", "exchange": "nasdaq", "pair": "NVDA"},
        "META": {"symbol": "NASDAQ:META", "exchange": "nasdaq", "pair": "META"},
        "SPY": {"symbol": "AMEX:SPY", "exchange": "amex", "pair": "SPY"},
    },
    "Commodities": {
        "GOLD": {"symbol": "TVC:GOLD", "exchange": "tvc", "pair": "GOLD"},
        "SILVER": {"symbol": "TVC:SILVER", "exchange": "tvc", "pair": "SILVER"},
        "OIL": {"symbol": "TVC:USOIL", "exchange": "tvc", "pair": "USOIL"},
        "NATGAS": {"symbol": "TVC:NATURALGAS", "exchange": "tvc", "pair": "NATGAS"},
    },
    "Currencies": {
        "EURUSD": {"symbol": "FX:EURUSD", "exchange": "fx", "pair": "EUR/USD"},
        "GBPUSD": {"symbol": "FX:GBPUSD", "exchange": "fx", "pair": "GBP/USD"},
        "USDJPY": {"symbol": "FX:USDJPY", "exchange": "fx", "pair": "USD/JPY"},
        "AUDUSD": {"symbol": "FX:AUDUSD", "exchange": "fx", "pair": "AUD/USD"},
    },
}

# Data directory
DATA_DIR = "Data"

# Default assets to collect (start with these)
DEFAULT_ASSETS = {
    "Crypto": ["BTC", "ETH"],
    "Stocks": ["SPY", "AAPL"],
    "Commodities": ["GOLD"],
    "Currencies": ["EURUSD"],
}
