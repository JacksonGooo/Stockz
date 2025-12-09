/**
 * Kraken Crypto Backfill
 *
 * Uses Kraken's FREE public API to get 1-minute historical data for crypto.
 * No API key required! Works in the US!
 *
 * Rate limit: Generous for public endpoints
 * Returns up to 720 candles per request
 *
 * Usage: node kraken_crypto_backfill.js [days]
 *        node kraken_crypto_backfill.js 365   - 1 year of crypto data
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Crypto');
const DAYS = parseInt(process.argv[2]) || 365;

// Map our asset names to Kraken pair names
// Focus on altcoins NOT well supported on Binance
const KRAKEN_PAIRS = {
    // These are the ones missing from Binance backfill
    'BCH': 'BCHUSD',        // Bitcoin Cash
    'ZEC': 'ZECUSD',        // Zcash
    'FIL': 'FILUSD',        // Filecoin
    'EOS': 'EOSUSD',        // EOS
    'XTZ': 'XTZUSD',        // Tezos
    'DASH': 'DASHUSD',      // Dash
    'XLM': 'XXLMZUSD',      // Stellar
    'ALGO': 'ALGOUSD',      // Algorand
    'ATOM': 'ATOMUSD',      // Cosmos
    'AAVE': 'AAVEUSD',      // Aave
    'MKR': 'MKRUSD',        // Maker
    'COMP': 'COMPUSD',      // Compound
    'SNX': 'SNXUSD',        // Synthetix
    'SUSHI': 'SUSHIUSD',    // Sushi
    'YFI': 'YFIUSD',        // Yearn
    'CRV': 'CRVUSD',        // Curve
    'BAL': 'BALUSD',        // Balancer
    'GRT': 'GRTUSD',        // The Graph
    'LRC': 'LRCUSD',        // Loopring
    'ENJ': 'ENJUSD',        // Enjin
    'MANA': 'MANAUSD',      // Decentraland
    'SAND': 'SANDUSD',      // Sandbox
    'AXS': 'AXSUSD',        // Axie Infinity
    'FLOW': 'FLOWUSD',      // Flow
    'KAVA': 'KAVAUSD',      // Kava
    'ANKR': 'ANKRUSD',      // Ankr
    'ENS': 'ENSUSD',        // ENS
    'THETA': 'THETAUSD',    // Theta
    'ICP': 'ICPUSD',        // Internet Computer
    'QNT': 'QNTUSD',        // Quant
    'CHZ': 'CHZUSD',        // Chiliz
    'ROSE': 'ROSEUSD',      // Oasis
    'APE': 'APEUSD',        // ApeCoin
};

// Assets to backfill (ones that Kraken supports)
const CRYPTO_ASSETS = Object.keys(KRAKEN_PAIRS);

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

// Fetch data from Kraken API
function fetchKrakenData(pair, since) {
    return new Promise((resolve, reject) => {
        // Kraken OHLC endpoint: interval=1 (1 minute)
        const url = `https://api.kraken.com/0/public/OHLC?pair=${pair}&interval=1&since=${since}`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    if (res.statusCode === 200) {
                        const parsed = JSON.parse(data);
                        if (parsed.error && parsed.error.length > 0) {
                            reject(new Error(parsed.error.join(', ')));
                        } else {
                            resolve(parsed);
                        }
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

// Convert Kraken OHLC format to our format
// Kraken format: [time, open, high, low, close, vwap, volume, count]
function convertKrakenCandle(ohlc) {
    return {
        timestamp: ohlc[0] * 1000, // Kraken uses seconds, we use milliseconds
        open: parseFloat(ohlc[1]),
        high: parseFloat(ohlc[2]),
        low: parseFloat(ohlc[3]),
        close: parseFloat(ohlc[4]),
        volume: parseFloat(ohlc[6])
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

// Group candles by date
function groupByDate(candles) {
    const groups = {};
    for (const candle of candles) {
        const date = new Date(candle.timestamp);
        const dateStr = formatDate(date);
        if (!groups[dateStr]) {
            groups[dateStr] = [];
        }
        groups[dateStr].push(candle);
    }
    return groups;
}

// Backfill an asset
async function backfillAsset(asset, days) {
    const krakenPair = KRAKEN_PAIRS[asset];
    if (!krakenPair) {
        console.log(`  Skipping ${asset} - no Kraken pair mapping`);
        return 0;
    }

    const assetDir = path.join(DATA_DIR, asset);
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - (days * 24 * 60 * 60);

    let totalNewCandles = 0;
    let currentSince = startTime;
    let allCandles = [];
    let requestCount = 0;
    const maxRequests = Math.ceil(days * 24 * 60 / 720) + 10; // Safety limit

    // Fetch all data in batches (720 candles = 12 hours per request)
    while (currentSince < endTime && requestCount < maxRequests) {
        try {
            requestCount++;
            const data = await fetchKrakenData(krakenPair, currentSince);

            // Find the result key (it varies by pair)
            const resultKey = Object.keys(data.result).find(k => k !== 'last');
            if (!resultKey || !data.result[resultKey]) {
                break;
            }

            const ohlcData = data.result[resultKey];
            if (ohlcData.length === 0) {
                break;
            }

            const candles = ohlcData.map(convertKrakenCandle);

            // Filter to our time range
            const filteredCandles = candles.filter(c =>
                c.timestamp >= startTime * 1000 && c.timestamp <= endTime * 1000
            );
            allCandles = allCandles.concat(filteredCandles);

            // Use Kraken's 'last' field for next pagination
            // This is the timestamp of the last returned candle
            const lastTimestamp = data.result.last;

            // If we got less than 720, or last is beyond our target, we're done
            if (ohlcData.length < 720 || lastTimestamp >= endTime) {
                break;
            }

            // Move to next batch - use lastTimestamp from Kraken
            currentSince = lastTimestamp;

            const progress = Math.min(100, ((currentSince - startTime) / (endTime - startTime) * 100)).toFixed(1);
            process.stdout.write(`\r    Fetched ${allCandles.length} candles (${progress}% of range)...    `);

            // Rate limiting - be nice to Kraken (1 second between requests)
            await sleep(1000);

        } catch (err) {
            if (err.message.includes('Unknown asset pair') || err.message.includes('Invalid asset pair')) {
                console.log(`\n    Pair ${krakenPair} not supported on Kraken`);
                return 0;
            }
            console.log(`\n    Error: ${err.message}`);
            // Wait and retry once
            await sleep(5000);
            continue;
        }
    }

    if (allCandles.length === 0) {
        console.log(`\n    No data received from Kraken`);
        return 0;
    }

    // Deduplicate candles by timestamp
    const uniqueCandles = [];
    const seen = new Set();
    for (const candle of allCandles) {
        if (!seen.has(candle.timestamp)) {
            seen.add(candle.timestamp);
            uniqueCandles.push(candle);
        }
    }
    allCandles = uniqueCandles.sort((a, b) => a.timestamp - b.timestamp);

    // Group candles by date and save to files
    const byDate = groupByDate(allCandles);
    const dates = Object.keys(byDate).sort();

    console.log(`\n    Got ${allCandles.length} candles across ${dates.length} days`);

    for (const dateStr of dates) {
        const date = new Date(dateStr + 'T00:00:00Z');
        const weekStart = getWeekStart(date);
        const weekStr = formatDate(weekStart);
        const weekFolder = `week_${weekStr}`;
        const filePath = path.join(assetDir, weekFolder, `${dateStr}.json`);

        const existing = loadJsonFile(filePath);
        const newCandles = byDate[dateStr];

        const merged = mergeCandles(existing, newCandles);
        const newCount = merged.length - existing.length;

        if (newCount > 0) {
            saveJsonFile(filePath, merged);
            totalNewCandles += newCount;
        }
    }

    console.log(`    Saved ${totalNewCandles} new candles across ${dates.length} days`);
    return totalNewCandles;
}

// Main
async function main() {
    console.log('='.repeat(60));
    console.log('Kraken Crypto Backfill (FREE - Works in US!)');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total assets: ${CRYPTO_ASSETS.length}`);
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (const asset of CRYPTO_ASSETS) {
        processed++;
        console.log(`\n[${processed}/${CRYPTO_ASSETS.length}] Processing Crypto/${asset}...`);

        try {
            const candles = await backfillAsset(asset, DAYS);
            totalCandles += candles;
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed || 1;
        const remaining = (CRYPTO_ASSETS.length - processed) / rate;
        console.log(`  Progress: ${(processed/CRYPTO_ASSETS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);

        // Longer delay between assets to be nice
        await sleep(2000);
    }

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
