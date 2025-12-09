/**
 * CryptoCompare Backfill for Altcoins
 *
 * Uses CryptoCompare's FREE histominute API for deep historical 1-minute data.
 * This fills in altcoins that Binance and Kraken don't fully support.
 *
 * API: https://min-api.cryptocompare.com/data/v2/histominute
 * Free tier: 100,000 calls/month
 * Returns up to 2000 candles per request
 *
 * Usage: node cryptocompare_backfill.js [days]
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Crypto');
const DAYS = parseInt(process.argv[2]) || 365;

// Altcoins that need CryptoCompare backfill
// (ones with limited Binance/Kraken support)
const ALTCOINS = [
    'BCH',      // Bitcoin Cash
    'ZEC',      // Zcash
    'FIL',      // Filecoin
    'EOS',      // EOS
    'XTZ',      // Tezos
    'DASH',     // Dash
    'XLM',      // Stellar
    'ALGO',     // Algorand
    'ATOM',     // Cosmos
    'AAVE',     // Aave
    'MKR',      // Maker
    'COMP',     // Compound
    'SNX',      // Synthetix
    'SUSHI',    // Sushi
    'YFI',      // Yearn
    'CRV',      // Curve
    'BAL',      // Balancer
    'GRT',      // The Graph
    'LRC',      // Loopring
    'ENJ',      // Enjin
    'MANA',     // Decentraland
    'SAND',     // Sandbox
    'AXS',      // Axie Infinity
    'FLOW',     // Flow
    'KAVA',     // Kava
    'ANKR',     // Ankr
    'ENS',      // ENS
    'THETA',    // Theta
    'ICP',      // Internet Computer
    'QNT',      // Quant
    'CHZ',      // Chiliz
    'ROSE',     // Oasis
    'APE',      // ApeCoin
    'HBAR',     // Hedera
    'VET',      // VeChain
    'EGLD',     // MultiversX
    'NEO',      // NEO
    'WAVES',    // Waves
];

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Get correct week start (Monday) using UTC
function getWeekStart(date) {
    const d = new Date(date);
    d.setUTCHours(0, 0, 0, 0);
    const day = d.getUTCDay();
    const diff = day === 0 ? -6 : 1 - day;
    d.setUTCDate(d.getUTCDate() + diff);
    return d;
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

// Fetch data from CryptoCompare API
function fetchCryptoCompare(symbol, toTs) {
    return new Promise((resolve, reject) => {
        // histominute returns up to 2000 candles ending at toTs
        const url = `https://min-api.cryptocompare.com/data/v2/histominute?fsym=${symbol}&tsym=USD&limit=2000&toTs=${toTs}`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    if (res.statusCode === 200) {
                        const parsed = JSON.parse(data);
                        if (parsed.Response === 'Success' && parsed.Data && parsed.Data.Data) {
                            resolve(parsed.Data);
                        } else if (parsed.Response === 'Error') {
                            reject(new Error(parsed.Message || 'API Error'));
                        } else {
                            resolve({ Data: [], TimeFrom: 0, TimeTo: 0 });
                        }
                    } else if (res.statusCode === 429) {
                        reject(new Error('RATE_LIMITED'));
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${data.substring(0, 200)}`));
                    }
                } catch (e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

// Convert CryptoCompare format to our format
function convertCandle(cc) {
    return {
        timestamp: cc.time * 1000,
        open: cc.open,
        high: cc.high,
        low: cc.low,
        close: cc.close,
        volume: cc.volumefrom
    };
}

// Load existing JSON file
function loadJsonFile(filePath) {
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
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
    for (const candle of existing) map.set(candle.timestamp, candle);
    for (const candle of newCandles) map.set(candle.timestamp, candle);
    return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

// Group candles by date
function groupByDate(candles) {
    const groups = {};
    for (const candle of candles) {
        const date = new Date(candle.timestamp);
        const dateStr = formatDate(date);
        if (!groups[dateStr]) groups[dateStr] = [];
        groups[dateStr].push(candle);
    }
    return groups;
}

// Backfill an asset
async function backfillAsset(symbol, days) {
    const assetDir = path.join(DATA_DIR, symbol);
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - (days * 24 * 60 * 60);

    let totalNewCandles = 0;
    let allCandles = [];
    let currentToTs = endTime;
    let requestCount = 0;
    const maxRequests = Math.ceil(days * 24 * 60 / 2000) + 10;

    // Fetch all data in batches (2000 candles = ~33 hours per request)
    while (currentToTs > startTime && requestCount < maxRequests) {
        try {
            requestCount++;
            const result = await fetchCryptoCompare(symbol, currentToTs);

            if (!result.Data || result.Data.length === 0) {
                break;
            }

            const candles = result.Data
                .filter(c => c.time >= startTime && c.time <= endTime)
                .filter(c => c.volumefrom > 0 || c.close > 0) // Skip empty candles
                .map(convertCandle);

            if (candles.length === 0) {
                break;
            }

            allCandles = allCandles.concat(candles);

            // Move to next batch - use TimeFrom from response
            currentToTs = result.TimeFrom - 1;

            const progress = Math.min(100, ((endTime - currentToTs) / (endTime - startTime) * 100)).toFixed(1);
            process.stdout.write(`\r    Fetched ${allCandles.length} candles (${progress}% of range)...    `);

            // Rate limiting - 50ms between requests (generous, CryptoCompare allows ~20/sec)
            await sleep(100);

        } catch (err) {
            if (err.message === 'RATE_LIMITED') {
                console.log(`\n    Rate limited, waiting 60s...`);
                await sleep(60000);
                continue;
            }
            if (err.message.includes('market does not exist')) {
                console.log(`\n    ${symbol} not available on CryptoCompare`);
                return 0;
            }
            console.log(`\n    Error: ${err.message}`);
            await sleep(5000);
            continue;
        }
    }

    if (allCandles.length === 0) {
        console.log(`\n    No data received from CryptoCompare`);
        return 0;
    }

    // Deduplicate
    const uniqueCandles = [];
    const seen = new Set();
    for (const candle of allCandles) {
        if (!seen.has(candle.timestamp)) {
            seen.add(candle.timestamp);
            uniqueCandles.push(candle);
        }
    }
    allCandles = uniqueCandles.sort((a, b) => a.timestamp - b.timestamp);

    // Group by date and save
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
    console.log('CryptoCompare Altcoin Backfill (FREE Deep History!)');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total altcoins: ${ALTCOINS.length}`);
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (const symbol of ALTCOINS) {
        processed++;
        console.log(`\n[${processed}/${ALTCOINS.length}] Processing ${symbol}...`);

        try {
            const candles = await backfillAsset(symbol, DAYS);
            totalCandles += candles;
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed || 1;
        const remaining = (ALTCOINS.length - processed) / rate;
        console.log(`  Progress: ${(processed/ALTCOINS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(1)}m | ETA: ${remaining.toFixed(1)}m`);

        // Delay between assets
        await sleep(1000);
    }

    const totalElapsed = (Date.now() - startTime) / 1000 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('CryptoCompare Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} minutes`);
    console.log(`Total new candles: ${totalCandles.toLocaleString()}`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
