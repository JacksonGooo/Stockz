/**
 * Alpaca Stocks Backfill - PARALLEL VERSION (3-4x faster!)
 *
 * Uses Alpaca's multi-symbol endpoint to fetch multiple stocks per request
 * Rate limit: 200 requests/minute
 *
 * Usage:
 *   node alpaca_stocks_parallel.js [days]
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data', 'Stock Market');
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');
const DAYS = parseInt(process.argv[2]) || 365;

const API_KEY = process.env.ALPACA_API_KEY || 'PKHEJ2BD4KXO7LSADIWYLU5JIQ';
const SECRET_KEY = process.env.ALPACA_SECRET_KEY || '8MyVxbmN3nUsCBD8TGWvXhtdjYmDEM6heTfvrTCkDQzE';

// Batch size: fetch this many symbols per request
const BATCH_SIZE = 5;
const REQUEST_DELAY = 350;

const STOCKS = [
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VXX', 'VIXY',
    'SQQQ', 'TQQQ', 'SPXS', 'SPXL',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MU', 'LRCX', 'AMAT', 'KLAC', 'MRVL', 'ASML', 'TSM',
    'ORCL', 'CRM', 'ADBE', 'NOW', 'PANW', 'CRWD', 'ZS', 'DDOG', 'SNOW', 'NET', 'PLTR', 'MDB', 'TEAM',
    'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC',
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD', 'MRNA', 'BIIB', 'REGN', 'VRTX',
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'DG', 'DLTR', 'TJX', 'ROST', 'CMG', 'YUM', 'DPZ',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'VLO', 'PSX',
    'NFLX', 'PYPL', 'SQ', 'COIN', 'UBER', 'ABNB', 'LYFT', 'SHOP', 'EBAY', 'ETSY', 'ZM', 'ROKU', 'SPOT', 'SNAP', 'PINS', 'TTD',
    'CAT', 'BA', 'UPS', 'FDX', 'HON', 'GE', 'RTX', 'LMT', 'NOC', 'GD', 'DE', 'MMM',
    'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'WBD', 'PARA',
    'GM', 'F', 'RIVN', 'LCID',
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
            source: 'Alpaca (Parallel)'
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
            source: 'Alpaca (Parallel)'
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

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

// Fetch MULTIPLE symbols at once
function fetchMultiSymbols(symbols, start, end, pageToken = null) {
    return new Promise((resolve, reject) => {
        const startISO = start.toISOString();
        const endISO = end.toISOString();
        const symbolsParam = symbols.join(',');

        let apiPath = `/v2/stocks/bars?symbols=${symbolsParam}&timeframe=1Min&start=${startISO}&end=${endISO}&limit=10000&adjustment=raw&feed=iex`;
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
                        resolve(JSON.parse(data));
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

function loadJsonFile(filePath) {
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch {
        return [];
    }
}

function saveJsonFile(filePath, data) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

function mergeCandles(existing, newCandles) {
    const map = new Map();
    for (const candle of existing) map.set(candle.timestamp, candle);
    for (const candle of newCandles) map.set(candle.timestamp, candle);
    return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

function saveCandlesByDay(symbol, candles) {
    const byDay = new Map();
    for (const candle of candles) {
        const dateStr = formatDate(new Date(candle.timestamp));
        if (!byDay.has(dateStr)) byDay.set(dateStr, []);
        byDay.get(dateStr).push(candle);
    }

    let savedCount = 0;
    for (const [dateStr, dayCandles] of byDay) {
        const weekStart = getWeekStart(new Date(dateStr));
        const weekFolder = `week_${formatDate(weekStart)}`;
        const filePath = path.join(DATA_DIR, symbol, weekFolder, `${dateStr}.json`);
        const merged = mergeCandles(loadJsonFile(filePath), dayCandles);
        saveJsonFile(filePath, merged);
        savedCount += dayCandles.length;
    }
    return savedCount;
}

// Backfill a batch of symbols
async function backfillBatch(symbols, days) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setUTCDate(startDate.getUTCDate() - days);

    const results = {};
    symbols.forEach(s => results[s] = 0);

    let nextPageToken = null;
    let retries = 0;

    do {
        try {
            const result = await fetchMultiSymbols(symbols, startDate, endDate, nextPageToken);

            // Process bars for each symbol
            if (result.bars) {
                for (const [symbol, bars] of Object.entries(result.bars)) {
                    if (bars && bars.length > 0) {
                        const candles = bars.map(convertAlpacaBar);
                        const saved = saveCandlesByDay(symbol, candles);
                        results[symbol] += saved;
                    }
                }
            }

            nextPageToken = result.next_page_token;
            retries = 0;
            await sleep(REQUEST_DELAY);
        } catch (err) {
            if (err.message === 'RATE_LIMITED') {
                retries++;
                if (retries > 3) break;
                console.log(`\n    Rate limited, waiting 60s...`);
                await sleep(60000);
                continue;
            }
            throw err;
        }
    } while (nextPageToken);

    return results;
}

async function main() {
    console.log('='.repeat(60));
    console.log('Alpaca Stocks Backfill - PARALLEL (3-4x faster!)');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total stocks: ${STOCKS.length}`);
    console.log(`Batch size: ${BATCH_SIZE} symbols per request`);
    console.log('');

    const batches = [];
    for (let i = 0; i < STOCKS.length; i += BATCH_SIZE) {
        batches.push(STOCKS.slice(i, i + BATCH_SIZE));
    }

    console.log(`Total batches: ${batches.length}`);
    console.log('Estimated time: 10-20 minutes');
    console.log('');
    console.log('Starting backfill...');
    console.log('='.repeat(60));

    const startTime = Date.now();
    let processed = 0;
    let totalCandles = 0;

    for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        const batchStr = batch.join(', ');
        console.log(`\n[Batch ${i + 1}/${batches.length}] ${batchStr}`);
        updateProgress(batchStr, processed, STOCKS.length, totalCandles);

        try {
            const results = await backfillBatch(batch, DAYS);

            for (const [symbol, count] of Object.entries(results)) {
                totalCandles += count;
                processed++;
                console.log(`  ${symbol}: ${count.toLocaleString()} candles`);
            }
        } catch (err) {
            console.log(`  Error: ${err.message}`);
            processed += batch.length;
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = (i + 1) / elapsed;
        const remaining = (batches.length - i - 1) / rate;
        console.log(`  Progress: ${(processed/STOCKS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(1)}m | ETA: ${remaining.toFixed(1)}m`);
        updateProgress(batchStr, processed, STOCKS.length, totalCandles);
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
