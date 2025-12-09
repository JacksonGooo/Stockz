/**
 * Backfill STOCKS ONLY via Polygon.io
 *
 * This works on the FREE tier! Crypto does NOT work on free tier.
 *
 * Usage: node backfill_stocks_only.js [days]
 *        node backfill_stocks_only.js 365   - 1 year of stock data
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const DAYS = parseInt(process.argv[2]) || 365;
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');

// Update progress file
function updateProgress(current, processed, total) {
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
        progress.stocks = {
            running: false,
            current: null,
            processed: 0,
            total: 0,
            lastUpdate: new Date().toISOString(),
            completed: true
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

// ALL stocks to backfill (works on Polygon free tier!)
const STOCKS = [
    // ETFs - Major Index
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VXX', 'VIXY',
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

console.log('='.repeat(60));
console.log('Polygon.io STOCKS Backfill (FREE TIER COMPATIBLE)');
console.log('='.repeat(60));
console.log(`Days to backfill: ${DAYS}`);
console.log(`Total stocks: ${STOCKS.length}`);
console.log('');

// Estimate time (12 seconds per API call, days calls per stock)
const estimatedHours = (STOCKS.length * DAYS * 12) / 3600;
console.log(`Estimated time: ${estimatedHours.toFixed(1)} hours`);
console.log('');
console.log('Starting backfill...');
console.log('='.repeat(60));

async function runBackfill(asset, days) {
    return new Promise((resolve, reject) => {
        const script = path.join(__dirname, 'polygon_backfill.js');
        const proc = spawn('node', [script, 'Stock Market', asset, days.toString()], {
            stdio: 'inherit',
            cwd: __dirname,
            env: { ...process.env, POLYGON_API_KEY: process.env.POLYGON_API_KEY }
        });

        proc.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`Backfill failed for ${asset}`));
            }
        });

        proc.on('error', reject);
    });
}

async function main() {
    let processed = 0;
    const startTime = Date.now();

    for (const stock of STOCKS) {
        processed++;
        console.log(`\n[${processed}/${STOCKS.length}] Processing Stock Market/${stock}...`);
        updateProgress(stock, processed, STOCKS.length);

        try {
            await runBackfill(stock, DAYS);
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (STOCKS.length - processed) / rate;
        console.log(`  Progress: ${(processed/STOCKS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);
        updateProgress(stock, processed, STOCKS.length);
    }

    markComplete();
    const totalElapsed = (Date.now() - startTime) / 1000 / 60 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('Stocks Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} hours`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
