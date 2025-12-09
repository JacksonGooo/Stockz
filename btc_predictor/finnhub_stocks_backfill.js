/**
 * Finnhub Stocks Backfill - 12x FASTER than Polygon!
 *
 * Finnhub offers:
 * - FREE tier with 60 API calls/minute (vs Polygon's 5/min)
 * - Stock candles with 1-minute resolution
 *
 * Sign up FREE at: https://finnhub.io/register
 * Get your API key from the dashboard
 *
 * Usage:
 *   set FINNHUB_API_KEY=your_key
 *   node finnhub_stocks_backfill.js [days]
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Stock Market');
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');
const DAYS = parseInt(process.argv[2]) || 365;

// Finnhub API key
const API_KEY = process.env.FINNHUB_API_KEY || 'd4qr289r01qrphabho0gd4qr289r01qrphabho10';

if (!API_KEY) {
    console.error('ERROR: Missing Finnhub API key!');
    console.error('');
    console.error('Sign up for FREE at: https://finnhub.io/register');
    console.error('Then set environment variable:');
    console.error('  set FINNHUB_API_KEY=your_key');
    console.error('');
    process.exit(1);
}

// Rate limiting: 60 requests per minute = 1000ms between requests (safe margin)
const REQUEST_DELAY = 1100; // ms between requests

// All stocks to backfill
const STOCKS = [
    // ETFs - Major Index
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
    'SQQQ', 'TQQQ', 'SPXS', 'SPXL',
    // Sector ETFs
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
    // Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    // Semiconductors
    'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MU', 'LRCX', 'AMAT', 'KLAC', 'MRVL', 'ASML', 'TSM',
    // Software & Cloud
    'ORCL', 'CRM', 'ADBE', 'NOW', 'PANW', 'CRWD', 'ZS', 'DDOG', 'SNOW', 'NET', 'PLTR', 'MDB', 'TEAM',
    // Finance
    'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC',
    // Healthcare & Biotech
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD', 'MRNA', 'BIIB', 'REGN', 'VRTX',
    // Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'DG', 'DLTR', 'TJX', 'ROST', 'CMG', 'YUM', 'DPZ',
    // Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'VLO', 'PSX',
    // Tech & Internet
    'NFLX', 'PYPL', 'SQ', 'COIN', 'UBER', 'ABNB', 'LYFT', 'SHOP', 'EBAY', 'ETSY', 'ZM', 'ROKU', 'SPOT', 'SNAP', 'PINS', 'TTD',
    // Industrial
    'CAT', 'BA', 'UPS', 'FDX', 'HON', 'GE', 'RTX', 'LMT', 'NOC', 'GD', 'DE', 'MMM',
    // Telecom & Media
    'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'WBD', 'PARA',
    // Auto
    'GM', 'F', 'RIVN', 'LCID',
    // Meme Stocks
    'GME', 'AMC', 'SOFI', 'HOOD', 'AFRM'
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
        progress.stocks = {
            running: true,
            current,
            processed,
            total,
            newCandles,
            lastUpdate: new Date().toISOString(),
            source: 'Finnhub'
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
        progress.stocks = {
            running: false,
            completed: true,
            newCandles: totalCandles,
            lastUpdate: new Date().toISOString(),
            source: 'Finnhub'
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

// Fetch stock candles from Finnhub
function fetchFinnhubData(symbol, from, to) {
    return new Promise((resolve, reject) => {
        const url = `https://finnhub.io/api/v1/stock/candle?symbol=${symbol}&resolution=1&from=${from}&to=${to}&token=${API_KEY}`;

        https.get(url, (res) => {
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
        }).on('error', reject);
    });
}

// Convert Finnhub candle format to our OHLCV format
function convertFinnhubCandles(data) {
    if (!data || data.s !== 'ok' || !data.t) {
        return [];
    }

    const candles = [];
    for (let i = 0; i < data.t.length; i++) {
        candles.push({
            timestamp: data.t[i] * 1000, // Convert to milliseconds
            open: data.o[i],
            high: data.h[i],
            low: data.l[i],
            close: data.c[i],
            volume: data.v[i]
        });
    }
    return candles;
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

// Backfill a single stock - fetch in weekly chunks
async function backfillStock(symbol, days) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setUTCDate(startDate.getUTCDate() - days);

    let totalCandles = 0;
    let currentStart = new Date(startDate);

    // Fetch in 7-day chunks (Finnhub has limits on data range)
    while (currentStart < endDate) {
        const currentEnd = new Date(currentStart);
        currentEnd.setUTCDate(currentEnd.getUTCDate() + 7);
        if (currentEnd > endDate) currentEnd.setTime(endDate.getTime());

        const fromTs = Math.floor(currentStart.getTime() / 1000);
        const toTs = Math.floor(currentEnd.getTime() / 1000);

        try {
            const result = await fetchFinnhubData(symbol, fromTs, toTs);
            const candles = convertFinnhubCandles(result);

            if (candles.length > 0) {
                const saved = saveCandlesByDay(symbol, candles);
                totalCandles += saved;
                process.stdout.write(`\r    Fetched ${totalCandles.toLocaleString()} candles...    `);
            }

            await sleep(REQUEST_DELAY);
        } catch (err) {
            if (err.message === 'RATE_LIMITED') {
                console.log(`\n    Rate limited, waiting 60s...`);
                await sleep(60000);
                continue; // Retry this chunk
            }
            // Skip other errors and continue
            console.log(`\n    Error: ${err.message}`);
        }

        currentStart = currentEnd;
    }

    console.log(`\n    Completed: ${totalCandles.toLocaleString()} candles`);
    return totalCandles;
}

async function main() {
    console.log('='.repeat(60));
    console.log('Finnhub Stocks Backfill (12x FASTER - 60 req/min!)');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total stocks: ${STOCKS.length}`);
    console.log('');

    // Estimate: ~52 requests per stock (365 days / 7 days per request), 60/min
    const requestsPerStock = Math.ceil(DAYS / 7);
    const totalRequests = STOCKS.length * requestsPerStock;
    const estimatedMinutes = (totalRequests * REQUEST_DELAY / 1000) / 60;
    console.log(`Estimated time: ${estimatedMinutes.toFixed(0)} minutes (~${(estimatedMinutes / 60).toFixed(1)} hours)`);
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (const symbol of STOCKS) {
        processed++;
        console.log(`\n[${processed}/${STOCKS.length}] Processing ${symbol}...`);
        updateProgress(symbol, processed, STOCKS.length, totalCandles);

        try {
            const candles = await backfillStock(symbol, DAYS);
            totalCandles += candles;
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (STOCKS.length - processed) / rate;
        console.log(`  Progress: ${(processed/STOCKS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(1)}m | ETA: ${remaining.toFixed(1)}m`);
        updateProgress(symbol, processed, STOCKS.length, totalCandles);
    }

    markComplete(totalCandles);
    const totalElapsed = (Date.now() - startTime) / 1000 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('Stocks Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} minutes`);
    console.log(`Total candles: ${totalCandles.toLocaleString()}`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
