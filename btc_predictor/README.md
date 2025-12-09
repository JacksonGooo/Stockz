# Bitcoin Price Predictor - BEAST MODE

**POWERFUL** neural network that predicts Bitcoin price 60 minutes into the future using historical candle data from Binance.

## Features

- ✅ Fetches 1 million candles of BTC/USDT data from **Binance** (highest liquidity!)
- ✅ **BEAST MODE LSTM** neural network (3.5M parameters!)
  - 5 LSTM layers: 512 → 256 → 256 → 128 → 64
  - 3 Dense layers: 128 → 64 → 32
  - Batch Normalization + Dropout
  - Early Stopping + Learning Rate Reduction
- ✅ **Compressed data caching** - Download once, use forever (10x smaller!)
- ✅ Technical indicators (RSI, Moving Averages, Volatility)
- ✅ **Interactive candlestick visualization**
- ✅ Predicts price 60 minutes ahead

## Setup

1. Install dependencies:
```bash
py -m pip install -r requirements.txt
```

2. Fetch data **(takes 20-30 minutes, but only do this ONCE!)**:
```bash
py fetch_data.py
```
This creates two files:
- `btc_1m_1M_candles.csv` (~150 MB) - Human readable
- `btc_1m_1M_candles.pkl.gz` (~15 MB) - **Compressed, FAST loading!**

The compressed file loads in seconds instead of minutes!

3. Train BEAST MODE model **(will take a while but pushes your PC!)**:
```bash
py train.py
```

4. Generate prediction visualization:
```bash
py visualize.py
```
Opens `btc_prediction.html` in your browser!

## Files

- `fetch_data.py` - Downloads Bitcoin historical data
- `model.py` - Neural network architecture
- `train.py` - Training and evaluation script
- `requirements.txt` - Python dependencies

## Model Architecture

- 3 LSTM layers (128 → 64 → 32 units)
- Dropout for regularization
- Dense output layer
- Uses 60 past candles to predict 60 minutes ahead

## Features Used

- OHLCV (Open, High, Low, Close, Volume)
- Returns and ratios
- Moving averages (7, 25, 99 period)
- RSI (Relative Strength Index)
- Volatility
- Volume ratios

## Output

- Trained model: `btc_predictor_60min.h5`
- Predictions plot: `predictions_plot.png`
- Training history: `training_history.png`

## Why Binance?

**Binance is the BEST data source for Bitcoin predictions:**
- ✅ #1 exchange by volume worldwide
- ✅ Highest liquidity = most accurate prices
- ✅ 24/7 trading, no gaps in data
- ✅ Free unlimited API access
- ✅ 1 MILLION+ candles available

Your model is learning from the **highest quality data possible!**

## Data Caching

Once you download data, it's saved in two formats:

1. **CSV** - Human readable, ~150 MB
2. **Pickle.gz** - Compressed, ~15 MB, **loads 10x faster!**

To re-download fresh data:
```bash
# Delete the cached files
del btc_1m_1M_candles.pkl.gz
del btc_1m_1M_candles.csv

# Then run fetch again
py fetch_data.py
```

## Notes

- This is for educational purposes
- Do NOT use for actual trading without extensive testing
- Market prediction is extremely difficult
- Past performance does not guarantee future results
- The BEAST MODE model has **3.5 MILLION parameters** - it will push your PC hard!
