/**
 * Alpaca Commodities (ETF) Backfill - FAST using Alpaca API
 *
 * Commodity ETFs trade as stocks, so Alpaca supports them!
 * - 200 requests/minute (vs Polygon's 5/min)
 * - Uses same API as stocks
 *
 * Usage:
 *   node alpaca_commodities_backfill.js [days]
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Commodities');
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');
const DAYS = parseInt(process.argv[2]) || 365;

// Alpaca API credentials (same as stocks)
const API_KEY = process.env.ALPACA_API_KEY || 'PKHEJ2BD4KXO7LSADIWYLU5JIQ';
const SECRET_KEY = process.env.ALPACA_SECRET_KEY || '8MyVxbmN3nUsCBD8TGWvXhtdjYmDEM6heTfvrTCkDQzE';

if (!API_KEY || !SECRET_KEY) {
    console.error('ERROR: Missing Alpaca API credentials!');
    process.exit(1);
}

// Rate limiting: 200 requests per minute = 300ms between requests
const REQUEST_DELAY = 350; // ms between requests (safe margin)

// Commodity ETFs to backfill
const COMMODITIES = [
    'DBC',   // Invesco DB Commodity Index
    'GSG',   // iShares S&P GSCI Commodity-Indexed Trust
    'GLD',   // Gold ETF
    'SLV',   // Silver ETF
    'USO',   // Oil ETF
    'UNG',   // Natural Gas ETF
    'CORN',  // Corn ETF (Teucrium)
    'WEAT',  // Wheat ETF
    'SOYB',  // Soybean ETF
    'CPER',  // Copper ETF
    'PPLT',  // Platinum ETF
    'PALL',  // Palladium ETF
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
        progress.commodities = {
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
        progress.commodities = {
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

// Fetch data from Alpaca API
function fetchAlpacaData(symbol, start, end, pageToken = null) {
    return new Promise((resolve, reject) => {
        const startISO = start.toISOString();
        const endISO = end.toISOString();

        let apiPath = `/v2/stocks/${symbol}/bars?timeframe=1Min&start=${startISO}&end=${endISO}&limit=10000&adjustment=raw&feed=iex`;
        if (pageToken) {
            apiPath += `&page_token=${pageToken}`;
        }

        const options = {
            hostname: 'data.alpaca.markets',
            path: apiPath,
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

// Convert Alpaca bar format to our OHLCV format
function convertAlpacaBar(bar) {
    return {
        timestamp: new Date(bar.t).getTime(),
        open: bar.o,
        high: bar.h,
        low: bar.l,
        close: bar.c,
        volume: bar.v
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

// Backfill a single commodity ETF
async function backfillCommodity(symbol, days) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setUTCDate(startDate.getUTCDate() - days);

    let totalCandles = 0;
    let nextPageToken = null;
    let retries = 0;

    // Alpaca returns data in pages, keep fetching until done
    do {
        try {
            const result = await fetchAlpacaData(symbol, startDate, endDate, nextPageToken);

            if (result.bars && result.bars.length > 0) {
                const candles = result.bars.map(convertAlpacaBar);
                const saved = saveCandlesByDay(symbol, candles);
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
            throw err;
        }
    } while (nextPageToken);

    console.log(`\n    Completed: ${totalCandles.toLocaleString()} candles`);
    return totalCandles;
}

async function main() {
    console.log('='.repeat(60));
    console.log('Alpaca Commodities (ETF) Backfill (FAST - 200 req/min!)');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total commodities: ${COMMODITIES.length}`);
    console.log('');

    const estimatedMinutes = (COMMODITIES.length * 5 * REQUEST_DELAY / 1000) / 60;
    console.log(`Estimated time: ${estimatedMinutes.toFixed(0)}-${(estimatedMinutes * 3).toFixed(0)} minutes`);
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (const symbol of COMMODITIES) {
        processed++;
        console.log(`\n[${processed}/${COMMODITIES.length}] Processing ${symbol}...`);
        updateProgress(symbol, processed, COMMODITIES.length, totalCandles);

        try {
            const candles = await backfillCommodity(symbol, DAYS);
            totalCandles += candles;
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (COMMODITIES.length - processed) / rate;
        console.log(`  Progress: ${(processed/COMMODITIES.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(1)}m | ETA: ${remaining.toFixed(1)}m`);
        updateProgress(symbol, processed, COMMODITIES.length, totalCandles);
    }

    markComplete(totalCandles);
    const totalElapsed = (Date.now() - startTime) / 1000 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('Commodities Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} minutes`);
    console.log(`Total candles: ${totalCandles.toLocaleString()}`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
