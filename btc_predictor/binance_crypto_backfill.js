/**
 * Binance Crypto Backfill
 *
 * Uses Binance's FREE public API to get 1-minute historical data for crypto.
 * No API key required!
 *
 * Rate limit: 1200 requests/min (very generous)
 * Max 1000 candles per request
 *
 * Usage: node binance_crypto_backfill.js [days]
 *        node binance_crypto_backfill.js 365   - 1 year of crypto data
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Crypto');
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');
const DAYS = parseInt(process.argv[2]) || 365;

// Update progress file
function updateProgress(current, processed, total, newCandles = 0) {
    try {
        let progress = {};
        if (fs.existsSync(PROGRESS_FILE)) {
            progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
        }
        progress.crypto = {
            running: true,
            current,
            processed,
            total,
            newCandles,
            lastUpdate: new Date().toISOString()
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {
        // Ignore progress file errors
    }
}

function markComplete() {
    try {
        let progress = {};
        if (fs.existsSync(PROGRESS_FILE)) {
            progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
        }
        progress.crypto = {
            running: false,
            current: null,
            processed: 0,
            total: 0,
            newCandles: 0,
            lastUpdate: new Date().toISOString(),
            completed: true
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

// All crypto assets to backfill (symbol without USDT suffix)
const CRYPTO_ASSETS = [
    'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT',
    'MATIC', 'LINK', 'LTC', 'UNI', 'ATOM', 'XLM', 'ALGO',
    'NEAR', 'FTM', 'SAND', 'MANA', 'AXS', 'AAVE', 'CRV',
    'SUSHI', 'COMP', '1INCH', 'ENJ', 'CHZ', 'BAT', 'ZRX'
];

// Map our asset names to Binance symbols
const SYMBOL_MAP = {
    'POL': 'MATIC',  // Polygon renamed
};

// Get correct week start (Monday) using UTC
function getWeekStart(date) {
    const d = new Date(date);
    const day = d.getUTCDay();
    const diff = d.getUTCDate() - day + (day === 0 ? -6 : 1);
    d.setUTCDate(diff);
    d.setUTCHours(0, 0, 0, 0);
    return d;
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Fetch data from Binance API
function fetchBinanceData(symbol, startTime, endTime) {
    return new Promise((resolve, reject) => {
        const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}USDT&interval=1m&limit=1000&startTime=${startTime}&endTime=${endTime}`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    if (res.statusCode === 200) {
                        const parsed = JSON.parse(data);
                        resolve(parsed);
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                    }
                } catch (e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

// Convert Binance kline format to our OHLCV format
function convertBinanceCandle(kline) {
    // Binance kline format:
    // [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume,
    // [6] Close time, [7] Quote asset volume, [8] Number of trades,
    // [9] Taker buy base asset volume, [10] Taker buy quote asset volume, [11] Ignore
    return {
        timestamp: kline[0],
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
        volume: parseFloat(kline[5])
    };
}

// Load existing JSON file
function loadJsonFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
    } catch {
        return [];
    }
}

// Save JSON file
function saveJsonFile(filePath, data) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

// Merge candles, deduplicating by timestamp
function mergeCandles(existing, newCandles) {
    const map = new Map();
    for (const candle of existing) {
        map.set(candle.timestamp, candle);
    }
    for (const candle of newCandles) {
        map.set(candle.timestamp, candle);
    }
    return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

// Backfill a single day for an asset
async function backfillDay(asset, date) {
    const binanceSymbol = SYMBOL_MAP[asset] || asset;

    // Calculate start and end of day in UTC
    const dayStart = new Date(date);
    dayStart.setUTCHours(0, 0, 0, 0);
    const dayEnd = new Date(date);
    dayEnd.setUTCHours(23, 59, 59, 999);

    const startMs = dayStart.getTime();
    const endMs = dayEnd.getTime();

    // We need to make multiple requests since limit is 1000 candles
    // and there are 1440 minutes in a day
    let allCandles = [];
    let currentStart = startMs;

    while (currentStart < endMs) {
        try {
            const data = await fetchBinanceData(binanceSymbol, currentStart, endMs);

            if (!data || data.length === 0) {
                break;
            }

            const candles = data.map(convertBinanceCandle);
            allCandles = allCandles.concat(candles);

            // Move to next batch (last candle timestamp + 1 minute)
            currentStart = data[data.length - 1][0] + 60000;

            // Small delay to be nice to the API
            await sleep(100);
        } catch (err) {
            if (err.message.includes('429')) {
                // Rate limited, wait and retry
                console.log(`    Rate limited, waiting 60s...`);
                await sleep(60000);
                continue;
            }
            throw err;
        }
    }

    return allCandles;
}

// Backfill an asset for specified number of days
async function backfillAsset(asset, days) {
    const assetDir = path.join(DATA_DIR, asset);

    const endDate = new Date();
    const startDate = new Date();
    startDate.setUTCDate(startDate.getUTCDate() - days);

    let totalCandles = 0;
    let daysProcessed = 0;

    // Process day by day
    const currentDate = new Date(startDate);

    while (currentDate <= endDate) {
        const dateStr = formatDate(currentDate);
        const weekStart = getWeekStart(currentDate);
        const weekStr = formatDate(weekStart);
        const weekFolder = `week_${weekStr}`;
        const filePath = path.join(assetDir, weekFolder, `${dateStr}.json`);

        // Check if we already have data for this day
        const existing = loadJsonFile(filePath);

        // Only backfill if we have less than 80% of expected candles (1440)
        if (existing.length < 1152) { // 80% of 1440
            try {
                const candles = await backfillDay(asset, currentDate);

                if (candles.length > 0) {
                    const merged = mergeCandles(existing, candles);
                    saveJsonFile(filePath, merged);
                    totalCandles += candles.length;
                    process.stdout.write(`\r    ${dateStr}: +${candles.length} candles (total: ${merged.length})    `);
                }
            } catch (err) {
                console.log(`\n    Error on ${dateStr}: ${err.message}`);
            }
        } else {
            process.stdout.write(`\r    ${dateStr}: already have ${existing.length} candles (skip)    `);
        }

        daysProcessed++;
        currentDate.setUTCDate(currentDate.getUTCDate() + 1);

        // Small delay between days
        await sleep(200);
    }

    console.log(`\n    Completed: ${totalCandles} new candles over ${daysProcessed} days`);
    return totalCandles;
}

// Main
async function main() {
    console.log('='.repeat(60));
    console.log('Binance Crypto Backfill (FREE - No API Key Required!)');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total assets: ${CRYPTO_ASSETS.length}`);
    console.log('');

    // Estimate time
    const estimatedMinutes = (CRYPTO_ASSETS.length * DAYS * 2 * 0.2) / 60; // ~0.2s per request, 2 requests per day
    console.log(`Estimated time: ${estimatedMinutes.toFixed(0)} minutes`);
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (const asset of CRYPTO_ASSETS) {
        processed++;
        console.log(`\n[${processed}/${CRYPTO_ASSETS.length}] Processing Crypto/${asset}...`);
        updateProgress(asset, processed, CRYPTO_ASSETS.length, totalCandles);

        try {
            const candles = await backfillAsset(asset, DAYS);
            totalCandles += candles;
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (CRYPTO_ASSETS.length - processed) / rate;
        console.log(`  Progress: ${(processed/CRYPTO_ASSETS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);
        updateProgress(asset, processed, CRYPTO_ASSETS.length, totalCandles);
    }

    markComplete();
    const totalElapsed = (Date.now() - startTime) / 1000 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('Crypto Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} minutes`);
    console.log(`Total new candles: ${totalCandles.toLocaleString()}`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
