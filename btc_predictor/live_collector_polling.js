/**
 * Live Data Collector with Polling - TradingView
 *
 * Polls TradingView every minute to get latest candle data.
 * This works around the limitation that TradingView doesn't stream
 * real-time data continuously on free tier.
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
        POL: 'COINBASE:POLUSD',
        LINK: 'COINBASE:LINKUSD',
        LTC: 'COINBASE:LTCUSD',
        UNI: 'COINBASE:UNIUSD',
        ATOM: 'COINBASE:ATOMUSD',
        XLM: 'COINBASE:XLMUSD',
        ALGO: 'COINBASE:ALGOUSD',
    },
    "Stock Market": {
        SPY: 'AMEX:SPY',
        QQQ: 'NASDAQ:QQQ',
        DIA: 'AMEX:DIA',
        IWM: 'AMEX:IWM',
        AAPL: 'NASDAQ:AAPL',
        MSFT: 'NASDAQ:MSFT',
        GOOGL: 'NASDAQ:GOOGL',
        AMZN: 'NASDAQ:AMZN',
        NVDA: 'NASDAQ:NVDA',
        META: 'NASDAQ:META',
        TSLA: 'NASDAQ:TSLA',
        JPM: 'NYSE:JPM',
        V: 'NYSE:V',
        MA: 'NYSE:MA',
        BAC: 'NYSE:BAC',
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
        NATGAS: 'NYMEX:NG1!',
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

const DATA_DIR = path.join(__dirname, '..', 'Data');
const POLL_INTERVAL = 60000; // Poll every 60 seconds

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
    const weekStart = getWeekStart(date);
    const assetDir = path.join(DATA_DIR, category, asset);
    const weekFolder = path.join(assetDir, `week_${formatDate(weekStart)}`);
    ensureDir(weekFolder);

    const dayFile = path.join(weekFolder, `${dayKey}.json`);
    let dayData = [];

    if (fs.existsSync(dayFile)) {
        dayData = JSON.parse(fs.readFileSync(dayFile, 'utf8'));
    }

    // Check if candle already exists (by timestamp)
    const existingIndex = dayData.findIndex(c => c.timestamp === candle.timestamp);
    if (existingIndex >= 0) {
        // Update existing candle with latest data
        dayData[existingIndex] = candle;
    } else {
        dayData.push(candle);
    }

    dayData.sort((a, b) => a.timestamp - b.timestamp);
    fs.writeFileSync(dayFile, JSON.stringify(dayData, null, 2));
    return existingIndex < 0; // true if new candle
}

// ============================================================
// Fetch data for a single asset
// ============================================================
function fetchAssetData(client, category, assetName, symbol) {
    return new Promise((resolve, reject) => {
        const chart = new client.Session.Chart();
        chart.setMarket(symbol, { timeframe: '1' });

        let resolved = false;
        const timeout = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                chart.delete();
                resolve({ success: false, error: 'timeout' });
            }
        }, 10000); // 10 second timeout per asset

        chart.onUpdate(() => {
            if (resolved) return;

            const periods = chart.periods;
            if (!periods || periods.length === 0) return;

            // Get all available candles (usually last few)
            let newCandles = 0;
            let lastPrice = 0;

            for (const period of periods) {
                const candle = {
                    timestamp: period.time * 1000,
                    open: period.open,
                    high: period.max,
                    low: period.min,
                    close: period.close,
                    volume: period.volume || 0
                };

                if (saveCandle(category, assetName, candle)) {
                    newCandles++;
                }
                lastPrice = candle.close;
            }

            clearTimeout(timeout);
            resolved = true;
            chart.delete();
            resolve({
                success: true,
                newCandles,
                lastPrice,
                totalPeriods: periods.length
            });
        });

        chart.onError((err) => {
            if (!resolved) {
                clearTimeout(timeout);
                resolved = true;
                chart.delete();
                resolve({ success: false, error: err });
            }
        });
    });
}

// ============================================================
// Poll all assets
// ============================================================
async function pollAllAssets() {
    const client = new TradingView.Client();
    const results = { success: 0, failed: 0, newCandles: 0 };

    console.log(`\n[${new Date().toISOString()}] Polling all assets...`);

    for (const [category, assets] of Object.entries(ASSETS)) {
        for (const [assetName, symbol] of Object.entries(assets)) {
            try {
                const result = await fetchAssetData(client, category, assetName, symbol);

                if (result.success) {
                    results.success++;
                    results.newCandles += result.newCandles;

                    if (result.newCandles > 0) {
                        console.log(`  ${category}/${assetName}: ${result.newCandles} new, last $${result.lastPrice.toFixed(2)}`);
                    }
                } else {
                    results.failed++;
                    if (result.error !== 'timeout') {
                        console.log(`  ${category}/${assetName}: Error - ${result.error}`);
                    }
                }
            } catch (err) {
                results.failed++;
                console.log(`  ${category}/${assetName}: Exception - ${err.message}`);
            }

            // Small delay between assets to avoid overwhelming
            await new Promise(r => setTimeout(r, 100));
        }
    }

    client.end();

    console.log(`Poll complete: ${results.success} success, ${results.failed} failed, ${results.newCandles} new candles`);
    return results;
}

// ============================================================
// Main Loop
// ============================================================
async function main() {
    console.log('============================================================');
    console.log('Live Data Collector (Polling Mode) - TradingView');
    console.log('============================================================');
    console.log(`Data directory: ${DATA_DIR}`);
    console.log(`Poll interval: ${POLL_INTERVAL / 1000} seconds`);
    console.log('============================================================');

    let totalAssets = 0;
    for (const [category, assets] of Object.entries(ASSETS)) {
        const assetList = Object.keys(assets).join(', ');
        console.log(`  ${category}: ${assetList}`);
        totalAssets += Object.keys(assets).length;
    }
    console.log(`\nTotal: ${totalAssets} assets`);
    console.log('============================================================');
    console.log('Press Ctrl+C to stop.\n');

    ensureDir(DATA_DIR);

    let totalPolls = 0;
    let totalNewCandles = 0;

    // Initial poll
    const initialResult = await pollAllAssets();
    totalPolls++;
    totalNewCandles += initialResult.newCandles;

    // Poll every POLL_INTERVAL
    const interval = setInterval(async () => {
        const result = await pollAllAssets();
        totalPolls++;
        totalNewCandles += result.newCandles;
    }, POLL_INTERVAL);

    // Graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n\nShutting down...');
        clearInterval(interval);
        console.log(`Total polls: ${totalPolls}`);
        console.log(`Total new candles: ${totalNewCandles}`);
        process.exit(0);
    });
}

main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
