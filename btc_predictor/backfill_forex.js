/**
 * Backfill FOREX/Currencies via Polygon.io
 * Uses C:EURUSD format for forex pairs
 *
 * Usage: node backfill_forex.js [days]
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const DAYS = parseInt(process.argv[2]) || 365;
const PROGRESS_FILE = path.join(__dirname, 'backfill_progress.json');

function updateProgress(current, processed, total) {
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
            lastUpdate: new Date().toISOString()
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

function markComplete() {
    try {
        let progress = {};
        if (fs.existsSync(PROGRESS_FILE)) {
            progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
        }
        progress.forex = {
            running: false,
            completed: true,
            lastUpdate: new Date().toISOString()
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

// All forex pairs (Polygon uses C:EURUSD format)
const FOREX_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD',
    'EURCHF', 'EURCAD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'AUDNZD',
    'AUDCAD', 'CADJPY', 'CHFJPY', 'NZDJPY', 'USDHKD', 'USDMXN',
    'USDNOK', 'USDSEK', 'USDTRY', 'USDZAR', 'USDHUF'
];

async function runBackfill(pair, days) {
    return new Promise((resolve, reject) => {
        const script = path.join(__dirname, 'polygon_backfill.js');
        // Use Currencies category with the pair name (polygon_backfill handles C: prefix)
        const proc = spawn('node', [script, 'Currencies', pair, days.toString()], {
            stdio: 'inherit',
            cwd: __dirname,
            env: { ...process.env }
        });

        proc.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`Backfill failed for ${pair}`));
        });
        proc.on('error', reject);
    });
}

async function main() {
    console.log('='.repeat(60));
    console.log('Polygon.io FOREX Backfill');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total pairs: ${FOREX_PAIRS.length}`);
    console.log('');

    const startTime = Date.now();
    let processed = 0;

    for (const pair of FOREX_PAIRS) {
        processed++;
        console.log(`\n[${processed}/${FOREX_PAIRS.length}] Processing Currencies/${pair}...`);
        updateProgress(pair, processed, FOREX_PAIRS.length);

        try {
            await runBackfill(pair, DAYS);
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (FOREX_PAIRS.length - processed) / rate;
        console.log(`  Progress: ${(processed/FOREX_PAIRS.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);
    }

    markComplete();
    console.log('\n' + '='.repeat(60));
    console.log('Forex Backfill Complete!');
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
