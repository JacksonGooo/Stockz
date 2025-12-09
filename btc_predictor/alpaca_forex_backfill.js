/**
 * Alpaca Forex Backfill - FAST using Alpaca API
 *
 * Alpaca supports forex through their market data API
 * - Uses /v1beta1/forex/bars endpoint
 * - 200 requests/minute
 *
 * Usage:
 *   node alpaca_forex_backfill.js [days]
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Currencies');
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');
const DAYS = parseInt(process.argv[2]) || 365;

// Alpaca API credentials
const API_KEY = process.env.ALPACA_API_KEY || 'PKHEJ2BD4KXO7LSADIWYLU5JIQ';
const SECRET_KEY = process.env.ALPACA_SECRET_KEY || '8MyVxbmN3nUsCBD8TGWvXhtdjYmDEM6heTfvrTCkDQzE';

if (!API_KEY || !SECRET_KEY) {
    console.error('ERROR: Missing Alpaca API credentials!');
    process.exit(1);
}

// Rate limiting: 200 requests per minute = 300ms between requests
const REQUEST_DELAY = 350;

// Forex pairs to backfill (Alpaca uses format like EUR/USD)
const FOREX_PAIRS = [
    { symbol: 'EURUSD', alpaca: 'EUR/USD' },
    { symbol: 'GBPUSD', alpaca: 'GBP/USD' },
    { symbol: 'USDJPY', alpaca: 'USD/JPY' },
    { symbol: 'USDCHF', alpaca: 'USD/CHF' },
    { symbol: 'AUDUSD', alpaca: 'AUD/USD' },
    { symbol: 'USDCAD', alpaca: 'USD/CAD' },
    { symbol: 'NZDUSD', alpaca: 'NZD/USD' },
    { symbol: 'EURGBP', alpaca: 'EUR/GBP' },
    { symbol: 'EURJPY', alpaca: 'EUR/JPY' },
    { symbol: 'GBPJPY', alpaca: 'GBP/JPY' },
    { symbol: 'AUDJPY', alpaca: 'AUD/JPY' },
    { symbol: 'EURAUD', alpaca: 'EUR/AUD' },
    { symbol: 'EURCHF', alpaca: 'EUR/CHF' },
    { symbol: 'EURCAD', alpaca: 'EUR/CAD' },
    { symbol: 'GBPAUD', alpaca: 'GBP/AUD' },
    { symbol: 'GBPCAD', alpaca: 'GBP/CAD' },
    { symbol: 'GBPCHF', alpaca: 'GBP/CHF' },
    { symbol: 'AUDNZD', alpaca: 'AUD/NZD' },
    { symbol: 'AUDCAD', alpaca: 'AUD/CAD' },
    { symbol: 'CADJPY', alpaca: 'CAD/JPY' },
    { symbol: 'CHFJPY', alpaca: 'CHF/JPY' },
    { symbol: 'NZDJPY', alpaca: 'NZD/JPY' },
    { symbol: 'USDHKD', alpaca: 'USD/HKD' },
    { symbol: 'USDMXN', alpaca: 'USD/MXN' },
    { symbol: 'USDNOK', alpaca: 'USD/NOK' },
    { symbol: 'USDSEK', alpaca: 'USD/SEK' },
    { symbol: 'USDTRY', alpaca: 'USD/TRY' },
    { symbol: 'USDZAR', alpaca: 'USD/ZAR' },
    { symbol: 'USDHUF', alpaca: 'USD/HUF' },
];

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function updateProgress(current, processed, total, newCandles = 0) {
    try {
        let progress = {};
        if (fs.existsSync(PROGRESS_FILE)) {
            progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
        }
        progress.forex = {
            running: true,
            current,
            processed,
            total,
            newCandles,
            lastUpdate: new Date().toISOString(),
            source: 'Alpaca'
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

function markComplete(totalCandles) {
    try {
        let progress = {};
        if (fs.existsSync(PROGRESS_FILE)) {
            progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
        }
        progress.forex = {
            running: false,
            completed: true,
            newCandles: totalCandles,
            lastUpdate: new Date().toISOString(),
            source: 'Alpaca'
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

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

// Fetch forex data from Alpaca API
function fetchAlpacaForex(alpacaSymbol, start, end) {
    return new Promise((resolve, reject) => {
        const startISO = start.toISOString();
        const endISO = end.toISOString();

        const options = {
            hostname: 'data.alpaca.markets',
            path: `/v1beta1/forex/bars?symbols=${encodeURIComponent(alpacaSymbol)}&timeframe=1Min&start=${startISO}&end=${endISO}&limit=10000`,
            method: 'GET',
            headers: {
                'APCA-API-KEY-ID': API_KEY,
                'APCA-API-SECRET-KEY': SECRET_KEY
            }
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    if (res.statusCode === 200) {
                        const parsed = JSON.parse(data);
                        resolve(parsed);
                    } else if (res.statusCode === 429) {
                        reject(new Error('RATE_LIMITED'));
                    } else if (res.statusCode === 403) {
                        reject(new Error('SUBSCRIPTION_REQUIRED'));
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${data.substring(0, 200)}`));
                    }
                } catch (e) {
                    reject(e);
                }
            });
        });

        req.on('error', reject);
        req.end();
    });
}

// Convert Alpaca forex bar format to our OHLCV format
function convertAlpacaBar(bar) {
    return {
        timestamp: new Date(bar.t).getTime(),
        open: bar.o,
        high: bar.h,
        low: bar.l,
        close: bar.c,
        volume: bar.v || 0
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

// Group candles by day and save to appropriate files
function saveCandlesByDay(symbol, candles) {
    const byDay = new Map();

    for (const candle of candles) {
        const date = new Date(candle.timestamp);
        const dateStr = formatDate(date);

        if (!byDay.has(dateStr)) {
            byDay.set(dateStr, []);
        }
        byDay.get(dateStr).push(candle);
    }

    let savedCount = 0;

    for (const [dateStr, dayCandles] of byDay) {
        const date = new Date(dateStr);
        const weekStart = getWeekStart(date);
        const weekStr = formatDate(weekStart);
        const weekFolder = `week_${weekStr}`;

        const filePath = path.join(DATA_DIR, symbol, weekFolder, `${dateStr}.json`);
        const existing = loadJsonFile(filePath);
        const merged = mergeCandles(existing, dayCandles);

        saveJsonFile(filePath, merged);
        savedCount += dayCandles.length;
    }

    return savedCount;
}

// Backfill a single forex pair
async function backfillForex(pair, days) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setUTCDate(startDate.getUTCDate() - days);

    let totalCandles = 0;
    let nextPageToken = null;
    let retries = 0;

    do {
        try {
            const result = await fetchAlpacaForex(pair.alpaca, startDate, endDate);

            // Alpaca forex returns data in format: { bars: { "EUR/USD": [...] } }
            const bars = result.bars && result.bars[pair.alpaca];
            if (bars && bars.length > 0) {
                const candles = bars.map(convertAlpacaBar);
                const saved = saveCandlesByDay(pair.symbol, candles);
                totalCandles += saved;
                process.stdout.write(`\r    Fetched ${totalCandles.toLocaleString()} candles...    `);
            }

            nextPageToken = result.next_page_token;
            retries = 0;

            await sleep(REQUEST_DELAY);
        } catch (err) {
            if (err.message === 'RATE_LIMITED') {
                retries++;
                if (retries > 3) {
                    console.log(`\n    Rate limited too many times, moving on`);
                    break;
                }
                console.log(`\n    Rate limited, waiting 60s...`);
                await sleep(60000);
                continue;
            }
            if (err.message === 'SUBSCRIPTION_REQUIRED') {
                console.log(`\n    Forex requires paid Alpaca subscription, skipping...`);
                return 0;
            }
            throw err;
        }
    } while (nextPageToken);

    console.log(`\n    Completed: ${totalCandles.toLocaleString()} candles`);
    return totalCandles;
}

async function main() {
    console.log('='.repeat(60));
    console.log('Alpaca Forex Backfill');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total pairs: ${FOREX_PAIRS.length}`);
    console.log('');
    console.log('NOTE: Forex may require paid Alpaca subscription.');
    console.log('If this fails, Polygon forex backfill is still running.');
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (const pair of FOREX_PAIRS) {
        processed++;
        console.log(`\n[${processed}/${FOREX_PAIRS.length}] Processing ${pair.symbol} (${pair.alpaca})...`);
        updateProgress(pair.symbol, processed, FOREX_PAIRS.length, totalCandles);

        try {
            const candles = await backfillForex(pair, DAYS);
            totalCandles += candles;

            // If first pair fails with subscription error, stop trying
            if (processed === 1 && candles === 0) {
                console.log('\nAlpaca forex requires paid subscription. Stopping.');
                console.log('The Polygon forex backfill is still running (slower but free).');
                break;
            }
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (FOREX_PAIRS.length - processed) / rate;
        console.log(`  Progress: ${(processed/FOREX_PAIRS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(1)}m | ETA: ${remaining.toFixed(1)}m`);
        updateProgress(pair.symbol, processed, FOREX_PAIRS.length, totalCandles);
    }

    markComplete(totalCandles);
    const totalElapsed = (Date.now() - startTime) / 1000 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('Forex Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} minutes`);
    console.log(`Total candles: ${totalCandles.toLocaleString()}`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
