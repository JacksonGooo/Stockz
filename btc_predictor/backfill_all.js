/**
 * Run Polygon Backfill for ALL Assets
 *
 * This runs polygon_backfill.js for each asset category with the specified days.
 *
 * Usage:
 *   node backfill_all.js [days]
 *   node backfill_all.js 730    - 2 years of data
 */

const { execSync, spawn } = require('child_process');
const path = require('path');

const DAYS = parseInt(process.argv[2]) || 730;

// Assets to backfill (matching polygon_backfill.js)
const ASSETS_TO_BACKFILL = {
    'Crypto': [
        'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT',
        'POL', 'LINK', 'LTC', 'UNI', 'ATOM', 'XLM', 'ALGO',
        'BCH', 'ETC', 'FIL', 'AAVE', 'MKR', 'COMP', 'SNX', 'SUSHI',
        'YFI', 'CRV', 'BAL', 'MATIC', 'SHIB', 'APE', 'ARB', 'OP'
    ],
    // Stocks and Forex may not work on free tier for 1-minute data
    // Uncomment to try:
    // 'Stock Market': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'],
};

console.log('='.repeat(60));
console.log('Polygon.io Mass Backfill');
console.log('='.repeat(60));
console.log(`Days to backfill: ${DAYS}`);
console.log('');

let totalAssets = 0;
for (const assets of Object.values(ASSETS_TO_BACKFILL)) {
    totalAssets += assets.length;
}
console.log(`Total assets to process: ${totalAssets}`);
console.log('');

// Calculate estimated time
const secondsPerDay = 12; // 12 sec per API call
const totalDays = totalAssets * DAYS;
const estimatedHours = (totalDays * secondsPerDay) / 3600;
console.log(`Estimated time: ${estimatedHours.toFixed(1)} hours (${(estimatedHours/24).toFixed(1)} days)`);
console.log('');
console.log('Starting backfill...');
console.log('='.repeat(60));
console.log('');

async function runBackfill(category, asset, days) {
    return new Promise((resolve, reject) => {
        const script = path.join(__dirname, 'polygon_backfill.js');
        const proc = spawn('node', [script, category, asset, days.toString()], {
            stdio: 'inherit',
            cwd: __dirname
        });

        proc.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`Backfill failed for ${category}/${asset}`));
            }
        });

        proc.on('error', reject);
    });
}

async function main() {
    let processed = 0;
    const startTime = Date.now();

    for (const [category, assets] of Object.entries(ASSETS_TO_BACKFILL)) {
        for (const asset of assets) {
            processed++;
            console.log(`\n[${processed}/${totalAssets}] Processing ${category}/${asset}...`);

            try {
                await runBackfill(category, asset, DAYS);
            } catch (err) {
                console.log(`  Error: ${err.message}`);
            }

            const elapsed = (Date.now() - startTime) / 1000 / 60;
            const rate = processed / elapsed;
            const remaining = (totalAssets - processed) / rate;
            console.log(`  Progress: ${(processed/totalAssets*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);
        }
    }

    const totalElapsed = (Date.now() - startTime) / 1000 / 60 / 60;
    console.log('\n' + '='.repeat(60));
    console.log('Mass Backfill Complete!');
    console.log(`Total time: ${totalElapsed.toFixed(1)} hours`);
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
