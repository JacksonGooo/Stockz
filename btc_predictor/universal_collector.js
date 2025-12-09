/**
 * Universal Live Data Collector - TradingView
 * Collects real-time data for multiple asset types:
 * - Crypto (BTC, ETH, etc.)
 * - Stocks (AAPL, GOOGL, etc.)
 * - Commodities (GOLD, SILVER, etc.)
 * - Currencies (EURUSD, GBPUSD, etc.)
 *
 * Saves to organized folder structure:
 * Data/
 *   Crypto/BTC/week_YYYY-MM-DD/YYYY-MM-DD.json
 *   Stocks/AAPL/week_YYYY-MM-DD/YYYY-MM-DD.json
 *   etc.
 */

const TradingView = require('@mathieuc/tradingview');
const fs = require('fs');
const path = require('path');

// ============================================================
// CONFIGURATION - ALL AVAILABLE ASSETS
// ============================================================
const ASSETS = {
    Crypto: {
        BTC: 'COINBASE:BTCUSD',
        ETH: 'COINBASE:ETHUSD',
        SOL: 'COINBASE:SOLUSD',
        XRP: 'COINBASE:XRPUSD',
        DOGE: 'COINBASE:DOGEUSD',
        ADA: 'COINBASE:ADAUSD',
        AVAX: 'COINBASE:AVAXUSD',
        DOT: 'COINBASE:DOTUSD',
        POL: 'COINBASE:POLUSD',  // Formerly MATIC
        LINK: 'COINBASE:LINKUSD',
        LTC: 'COINBASE:LTCUSD',
        UNI: 'COINBASE:UNIUSD',
        ATOM: 'COINBASE:ATOMUSD',
        XLM: 'COINBASE:XLMUSD',
        ALGO: 'COINBASE:ALGOUSD',
    },
    "Stock Market": {
        // Major Indices ETFs
        SPY: 'AMEX:SPY',
        QQQ: 'NASDAQ:QQQ',
        DIA: 'AMEX:DIA',
        IWM: 'AMEX:IWM',
        // Tech Giants
        AAPL: 'NASDAQ:AAPL',
        MSFT: 'NASDAQ:MSFT',
        GOOGL: 'NASDAQ:GOOGL',
        AMZN: 'NASDAQ:AMZN',
        NVDA: 'NASDAQ:NVDA',
        META: 'NASDAQ:META',
        TSLA: 'NASDAQ:TSLA',
        // Finance
        JPM: 'NYSE:JPM',
        V: 'NYSE:V',
        MA: 'NYSE:MA',
        BAC: 'NYSE:BAC',
        // Other Major
        UNH: 'NYSE:UNH',
        JNJ: 'NYSE:JNJ',
        WMT: 'NYSE:WMT',
        PG: 'NYSE:PG',
        XOM: 'NYSE:XOM',
    },
    Commodities: {
        GOLD: 'TVC:GOLD',
        SILVER: 'TVC:SILVER',
        OIL: 'TVC:USOIL',
        NATGAS: 'NYMEX:NG1!',  // Natural Gas Futures
        COPPER: 'COMEX:HG1!',
        PLATINUM: 'TVC:PLATINUM',
    },
    Currencies: {
        EURUSD: 'FX:EURUSD',
        GBPUSD: 'FX:GBPUSD',
        USDJPY: 'FX:USDJPY',
        AUDUSD: 'FX:AUDUSD',
        USDCAD: 'FX:USDCAD',
        USDCHF: 'FX:USDCHF',
        NZDUSD: 'FX:NZDUSD',
        EURGBP: 'FX:EURGBP',
    },
};

const DATA_DIR = path.join(__dirname, '..', 'Data');  // Root Stockz/Data folder

// ============================================================
// Helper Functions
// ============================================================
function getWeekStart(date) {
    const d = new Date(date);
    const day = d.getUTCDay();
    const diff = d.getUTCDate() - day + (day === 0 ? -6 : 1);
    d.setUTCDate(diff);
    return d;
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function formatTime(date) {
    return date.toISOString().replace('T', ' ').split('.')[0];
}

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

function saveCandle(category, asset, candle) {
    const date = new Date(candle.timestamp);
    const dayKey = formatDate(date);

    // All data goes to root Stockz/Data folder
    const baseDir = DATA_DIR;

    // Build path: Data/Category/Asset/week_YYYY-MM-DD/YYYY-MM-DD.json
    const weekStart = getWeekStart(date);
    const assetDir = path.join(baseDir, category, asset);
    const weekFolder = path.join(assetDir, `week_${formatDate(weekStart)}`);
    ensureDir(weekFolder);

    const dayFile = path.join(weekFolder, `${dayKey}.json`);
    let dayData = [];

    if (fs.existsSync(dayFile)) {
        dayData = JSON.parse(fs.readFileSync(dayFile, 'utf8'));
    }

    // Check if candle already exists
    const exists = dayData.some(c => c.timestamp === candle.timestamp);
    if (!exists) {
        dayData.push(candle);
        dayData.sort((a, b) => a.timestamp - b.timestamp);
        fs.writeFileSync(dayFile, JSON.stringify(dayData, null, 2));
        return true;
    }
    return false;
}

// ============================================================
// Main Collector
// ============================================================
async function startUniversalCollector() {
    console.log('============================================================');
    console.log('Universal Live Data Collector - TradingView');
    console.log('============================================================');
    console.log(`Data directory: ${DATA_DIR}`);
    console.log('============================================================');
    console.log('Assets being tracked:');

    let totalAssets = 0;
    for (const [category, assets] of Object.entries(ASSETS)) {
        const assetList = Object.keys(assets).join(', ');
        console.log(`  ${category}: ${assetList}`);
        totalAssets += Object.keys(assets).length;
    }

    console.log(`\nTotal: ${totalAssets} assets`);
    console.log('============================================================');
    console.log('Collecting real-time data. Press Ctrl+C to stop.');
    console.log('============================================================\n');

    ensureDir(DATA_DIR);

    const client = new TradingView.Client();
    const charts = {};
    const stats = {};
    let startTime = Date.now();

    // Create a chart for each asset
    for (const [category, assets] of Object.entries(ASSETS)) {
        for (const [assetName, symbol] of Object.entries(assets)) {
            const key = `${category}:${assetName}`;
            stats[key] = { candles: 0, lastPrice: 0 };

            const chart = new client.Session.Chart();
            chart.setMarket(symbol, { timeframe: '1' });

            let lastCandle = null;

            chart.onUpdate(() => {
                const periods = chart.periods;
                if (!periods || periods.length === 0) return;

                const latest = periods[periods.length - 1];

                if (lastCandle === null || latest.time !== lastCandle.time) {
                    const candle = {
                        timestamp: latest.time * 1000,
                        open: latest.open,
                        high: latest.max,
                        low: latest.min,
                        close: latest.close,
                        volume: latest.volume || 0
                    };

                    const saved = saveCandle(category, assetName, candle);
                    if (saved) {
                        stats[key].candles++;
                        stats[key].lastPrice = candle.close;

                        const date = new Date(candle.timestamp);
                        console.log(`[${formatTime(date)}] ${category}/${assetName}: $${candle.close.toFixed(2)}`);
                    }

                    lastCandle = latest;
                }
            });

            chart.onError((err) => {
                console.log(`Error ${key}:`, err);
            });

            charts[key] = chart;
        }
    }

    console.log('Waiting for data from all assets...\n');

    // Status update every minute
    setInterval(() => {
        const runtime = Math.floor((Date.now() - startTime) / 1000 / 60);
        console.log('\n--- Status Update ---');
        console.log(`Runtime: ${runtime} minutes`);

        for (const [key, stat] of Object.entries(stats)) {
            if (stat.candles > 0) {
                console.log(`  ${key}: ${stat.candles} candles, last: $${stat.lastPrice.toFixed(2)}`);
            }
        }
        console.log('---------------------\n');
    }, 60000);

    // Graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n\nShutting down...');
        console.log('Final stats:');
        for (const [key, stat] of Object.entries(stats)) {
            console.log(`  ${key}: ${stat.candles} candles collected`);
        }
        client.end();
        process.exit(0);
    });

    // Keep running
    await new Promise(() => {});
}

startUniversalCollector().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
