/**
 * Polygon.io Historical Data Fetcher
 * Fetches 2 years of 1-minute data for stocks, commodities, currencies
 * Uses same OHLCV format as Coinbase/TradingView
 *
 * Get free API key: https://polygon.io/dashboard/signup
 * Free tier: 5 calls/minute, 2 years of minute data
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// API Key - set via environment variable or replace here
const API_KEY = process.env.POLYGON_API_KEY || 'YOUR_API_KEY_HERE';

// Data directory
const DATA_DIR = path.join(__dirname, '..', 'Data');

// Asset mappings to Polygon symbols
const ASSETS = {
  'Stock Market': {
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'DIA': 'DIA',
    'IWM': 'IWM',
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'GOOGL': 'GOOGL',
    'AMZN': 'AMZN',
    'NVDA': 'NVDA',
    'META': 'META',
    'TSLA': 'TSLA',
    'JPM': 'JPM',
    'V': 'V',
    'MA': 'MA',
    'BAC': 'BAC',
    'UNH': 'UNH',
    'JNJ': 'JNJ',
    'WMT': 'WMT',
    'PG': 'PG',
    'XOM': 'XOM',
  },
  // Note: Polygon free tier mainly supports stocks
  // Commodities and currencies need paid plans
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

// Save candles to file (same format as Coinbase/TradingView)
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

// Fetch data from Polygon API
function fetchPolygon(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

// Fetch one day of minute data
async function fetchDay(symbol, date) {
  const url = `https://api.polygon.io/v2/aggs/ticker/${symbol}/range/1/minute/${date}/${date}?adjusted=true&sort=asc&limit=50000&apiKey=${API_KEY}`;

  try {
    const data = await fetchPolygon(url);

    if (data.status === 'ERROR') {
      console.log(`    Error: ${data.error || 'Unknown error'}`);
      return [];
    }

    if (!data.results || data.results.length === 0) {
      return [];
    }

    // Convert to our OHLCV format (same as Coinbase/TradingView)
    return data.results.map(bar => ({
      timestamp: bar.t,  // Already in milliseconds
      open: bar.o,
      high: bar.h,
      low: bar.l,
      close: bar.c,
      volume: bar.v || 0,
    }));
  } catch (error) {
    console.log(`    Fetch error: ${error.message}`);
    return [];
  }
}

// Fetch historical data for one asset
async function fetchAsset(category, assetName, polygonSymbol, daysBack = 730) {
  console.log(`\n[${category}/${assetName}] Fetching ${daysBack} days from Polygon.io...`);

  let totalCandles = 0;
  let totalDays = 0;
  const today = new Date();

  for (let i = 0; i < daysBack; i++) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const dateStr = formatDate(date);

    // Skip weekends for stocks
    const dayOfWeek = date.getDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) continue;

    // Check if we already have this day's data
    const weekStart = getWeekStart(date);
    const weekDir = path.join(DATA_DIR, category, assetName, `week_${formatDate(weekStart)}`);
    const filePath = path.join(weekDir, `${dateStr}.json`);

    if (fs.existsSync(filePath)) {
      try {
        const existing = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        if (existing.length > 300) {  // Already have substantial data
          continue;
        }
      } catch (e) {}
    }

    // Fetch this day
    const candles = await fetchDay(polygonSymbol, dateStr);

    if (candles.length > 0) {
      saveCandles(category, assetName, dateStr, candles);
      totalCandles += candles.length;
      totalDays++;

      if (totalDays % 10 === 0) {
        console.log(`  Progress: ${totalDays} days, ${totalCandles.toLocaleString()} candles`);
      }
    }

    // Rate limit: 5 calls/minute for free tier
    await new Promise(r => setTimeout(r, 12500));  // ~5 calls per minute
  }

  console.log(`  Completed: ${totalDays} days, ${totalCandles.toLocaleString()} candles`);
  return totalCandles;
}

// Main function
async function main() {
  if (API_KEY === 'YOUR_API_KEY_HERE') {
    console.log('============================================================');
    console.log('Polygon.io Historical Data Fetcher');
    console.log('============================================================');
    console.log('');
    console.log('ERROR: No API key set!');
    console.log('');
    console.log('Get a free API key at: https://polygon.io/dashboard/signup');
    console.log('');
    console.log('Then run with:');
    console.log('  set POLYGON_API_KEY=your_key_here');
    console.log('  node fetch_polygon_historical.js');
    console.log('');
    console.log('Or edit this file and replace YOUR_API_KEY_HERE');
    console.log('============================================================');
    return;
  }

  const args = process.argv.slice(2);
  const specificCategory = args[0];
  const specificAsset = args[1];
  const daysBack = parseInt(args[2]) || 730;

  console.log('============================================================');
  console.log('Polygon.io Historical Data Fetcher');
  console.log('============================================================');
  console.log(`Data directory: ${DATA_DIR}`);
  console.log(`Fetching ${daysBack} days of 1-minute data`);
  console.log('Format: Same OHLCV as Coinbase/TradingView');
  console.log('Rate limit: 5 calls/minute (free tier)');
  console.log('============================================================');

  let totalCandles = 0;
  let totalAssets = 0;

  for (const [category, assets] of Object.entries(ASSETS)) {
    if (specificCategory && category !== specificCategory) continue;

    console.log(`\n========== ${category} ==========`);

    for (const [assetName, polygonSymbol] of Object.entries(assets)) {
      if (specificAsset && assetName !== specificAsset) continue;

      const candles = await fetchAsset(category, assetName, polygonSymbol, daysBack);
      totalCandles += candles;
      totalAssets++;
    }
  }

  console.log('\n============================================================');
  console.log(`Completed! Fetched ${totalCandles.toLocaleString()} candles for ${totalAssets} assets`);
  console.log('============================================================');
}

// Run
main().catch(console.error);
