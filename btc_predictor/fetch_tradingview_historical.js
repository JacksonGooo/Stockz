/**
 * TradingView Historical Data Fetcher
 * Fetches historical 1-minute candles for stocks, commodities, currencies
 * Uses same OHLCV format as Coinbase crypto data
 */

const TradingView = require('@mathieuc/tradingview');
const fs = require('fs');
const path = require('path');

// Data directory
const DATA_DIR = path.join(__dirname, '..', 'Data');

// Asset configurations (same as live collector)
const ASSETS = {
  'Stock Market': {
    'SPY': 'AMEX:SPY',
    'QQQ': 'NASDAQ:QQQ',
    'DIA': 'AMEX:DIA',
    'IWM': 'AMEX:IWM',
    'AAPL': 'NASDAQ:AAPL',
    'MSFT': 'NASDAQ:MSFT',
    'GOOGL': 'NASDAQ:GOOGL',
    'AMZN': 'NASDAQ:AMZN',
    'NVDA': 'NASDAQ:NVDA',
    'META': 'NASDAQ:META',
    'TSLA': 'NASDAQ:TSLA',
    'JPM': 'NYSE:JPM',
    'V': 'NYSE:V',
    'MA': 'NYSE:MA',
    'BAC': 'NYSE:BAC',
    'UNH': 'NYSE:UNH',
    'JNJ': 'NYSE:JNJ',
    'WMT': 'NYSE:WMT',
    'PG': 'NYSE:PG',
    'XOM': 'NYSE:XOM',
  },
  'Commodities': {
    'GOLD': 'TVC:GOLD',
    'SILVER': 'TVC:SILVER',
    'OIL': 'TVC:USOIL',
    'NATGAS': 'TVC:NATGAS',
    'COPPER': 'COMEX:HG1!',
    'PLATINUM': 'TVC:PLATINUM',
  },
  'Currencies': {
    'EURUSD': 'FX:EURUSD',
    'GBPUSD': 'FX:GBPUSD',
    'USDJPY': 'FX:USDJPY',
    'AUDUSD': 'FX:AUDUSD',
    'USDCAD': 'FX:USDCAD',
    'USDCHF': 'FX:USDCHF',
    'NZDUSD': 'FX:NZDUSD',
    'EURGBP': 'FX:EURGBP',
  },
  'Crypto': {
    'BTC': 'COINBASE:BTCUSD',
    'ETH': 'COINBASE:ETHUSD',
    'SOL': 'COINBASE:SOLUSD',
    'XRP': 'COINBASE:XRPUSD',
    'DOGE': 'COINBASE:DOGEUSD',
    'ADA': 'COINBASE:ADAUSD',
    'AVAX': 'COINBASE:AVAXUSD',
    'DOT': 'COINBASE:DOTUSD',
    'POL': 'COINBASE:POLUSD',
    'LINK': 'COINBASE:LINKUSD',
    'LTC': 'COINBASE:LTCUSD',
    'UNI': 'COINBASE:UNIUSD',
    'ATOM': 'COINBASE:ATOMUSD',
    'XLM': 'COINBASE:XLMUSD',
    'ALGO': 'COINBASE:ALGOUSD',
  },
};

// Helper to get week start date (Monday)
function getWeekStart(date) {
  const d = new Date(date);
  const day = d.getUTCDay();
  const diff = d.getUTCDate() - day + (day === 0 ? -6 : 1);
  d.setUTCDate(diff);
  return d;
}

// Format date as YYYY-MM-DD
function formatDate(date) {
  return date.toISOString().split('T')[0];
}

// Ensure directory exists
function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// Save candles to file (same format as Coinbase)
function saveCandles(category, asset, date, candles) {
  const weekStart = getWeekStart(new Date(date));
  const weekDir = path.join(DATA_DIR, category, asset, `week_${formatDate(weekStart)}`);
  ensureDir(weekDir);

  const filePath = path.join(weekDir, `${date}.json`);

  // If file exists, merge with existing data
  let existingCandles = [];
  if (fs.existsSync(filePath)) {
    try {
      existingCandles = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (e) {
      existingCandles = [];
    }
  }

  // Merge and deduplicate by timestamp
  const allCandles = [...existingCandles, ...candles];
  const uniqueCandles = Array.from(
    new Map(allCandles.map(c => [c.timestamp, c])).values()
  ).sort((a, b) => a.timestamp - b.timestamp);

  fs.writeFileSync(filePath, JSON.stringify(uniqueCandles, null, 2));
  return uniqueCandles.length;
}

// Fetch historical data for one asset
async function fetchAssetHistorical(client, category, assetName, tvSymbol, daysBack = 730) {
  return new Promise((resolve) => {
    console.log(`\n[${category}/${assetName}] Fetching ${daysBack} days from TradingView...`);

    const chart = new client.Session.Chart();
    chart.setMarket(tvSymbol, {
      timeframe: '1',  // 1 minute
      range: daysBack * 1440,  // days * minutes per day (request max candles)
    });

    let resolved = false;
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        console.log(`  Timeout for ${assetName}`);
        chart.delete();
        resolve(0);
      }
    }, 60000); // 60 second timeout

    chart.onSymbolLoaded(() => {
      console.log(`  Symbol loaded: ${tvSymbol}`);
    });

    chart.onUpdate(() => {
      if (resolved) return;

      const periods = chart.periods;
      if (!periods || periods.length === 0) {
        return;
      }

      resolved = true;
      clearTimeout(timeout);

      console.log(`  Received ${periods.length} candles`);

      // Convert to our OHLCV format (same as Coinbase)
      const candles = periods.map(p => ({
        timestamp: p.time * 1000,  // Convert to milliseconds
        open: p.open,
        high: p.max,
        low: p.min,
        close: p.close,
        volume: p.volume || 0,
      })).filter(c => c.open && c.high && c.low && c.close);

      // Group by date and save
      const candlesByDate = {};
      for (const candle of candles) {
        const date = formatDate(new Date(candle.timestamp));
        if (!candlesByDate[date]) {
          candlesByDate[date] = [];
        }
        candlesByDate[date].push(candle);
      }

      let totalSaved = 0;
      for (const [date, dayCandles] of Object.entries(candlesByDate)) {
        const count = saveCandles(category, assetName, date, dayCandles);
        totalSaved += dayCandles.length;
      }

      console.log(`  Saved ${totalSaved} candles across ${Object.keys(candlesByDate).length} days`);

      chart.delete();
      resolve(totalSaved);
    });

    chart.onError((err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        console.log(`  Error for ${assetName}: ${err}`);
        chart.delete();
        resolve(0);
      }
    });
  });
}

// Main function
async function main() {
  const args = process.argv.slice(2);
  const specificCategory = args[0];
  const specificAsset = args[1];
  const daysBack = parseInt(args[2]) || 730; // Default 2 years

  console.log('============================================================');
  console.log('TradingView Historical Data Fetcher');
  console.log('============================================================');
  console.log(`Data directory: ${DATA_DIR}`);
  console.log(`Fetching ${daysBack} days of 1-minute data`);
  console.log('Format: Same OHLCV as Coinbase (timestamp, open, high, low, close, volume)');
  console.log('============================================================');

  const client = new TradingView.Client();
  let totalCandles = 0;
  let totalAssets = 0;

  try {
    for (const [category, assets] of Object.entries(ASSETS)) {
      if (specificCategory && category !== specificCategory) continue;

      console.log(`\n========== ${category} ==========`);

      for (const [assetName, tvSymbol] of Object.entries(assets)) {
        if (specificAsset && assetName !== specificAsset) continue;

        const candles = await fetchAssetHistorical(client, category, assetName, tvSymbol, daysBack);
        totalCandles += candles;
        totalAssets++;

        // Small delay between assets
        await new Promise(r => setTimeout(r, 2000));
      }
    }
  } finally {
    client.end();
  }

  console.log('\n============================================================');
  console.log(`Completed! Fetched ${totalCandles.toLocaleString()} candles for ${totalAssets} assets`);
  console.log('============================================================');
}

// Run
main().catch(console.error);
