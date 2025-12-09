/**
 * Backfill COMMODITIES via Polygon.io
 * Some are ETFs (DBC, GSG), others may need different handling
 *
 * Usage: node backfill_commodities.js [days]
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
        progress.commodities = {
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
        progress.commodities = {
            running: false,
            completed: true,
            lastUpdate: new Date().toISOString()
        };
        fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
    } catch (e) {}
}

// Commodity ETFs and futures
// ETFs like DBC, GSG work as stocks
// For metals/energy, we'll try stock-style first (some trade as ETFs)
const COMMODITIES = [
    // ETFs
    'DBC',   // Invesco DB Commodity Index
    'GSG',   // iShares S&P GSCI Commodity-Indexed Trust
    // Individual commodity ETFs
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

async function runBackfill(asset, days) {
    return new Promise((resolve, reject) => {
        const script = path.join(__dirname, 'polygon_backfill.js');
        // Commodities stored in Stock Market style for ETFs
        const proc = spawn('node', [script, 'Commodities', asset, days.toString()], {
            stdio: 'inherit',
            cwd: __dirname,
            env: { ...process.env }
        });

        proc.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`Backfill failed for ${asset}`));
        });
        proc.on('error', reject);
    });
}

async function main() {
    console.log('='.repeat(60));
    console.log('Polygon.io COMMODITIES Backfill');
    console.log('='.repeat(60));
    console.log(`Days to backfill: ${DAYS}`);
    console.log(`Total assets: ${COMMODITIES.length}`);
    console.log('');

    const startTime = Date.now();
    let processed = 0;

    for (const asset of COMMODITIES) {
        processed++;
        console.log(`\n[${processed}/${COMMODITIES.length}] Processing Commodities/${asset}...`);
        updateProgress(asset, processed, COMMODITIES.length);

        try {
            await runBackfill(asset, DAYS);
        } catch (err) {
            console.log(`  Error: ${err.message}`);
        }

        const elapsed = (Date.now() - startTime) / 1000 / 60;
        const rate = processed / elapsed;
        const remaining = (COMMODITIES.length - processed) / rate;
        console.log(`  Progress: ${(processed/COMMODITIES.length*100).toFixed(1)}% | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);
    }

    markComplete();
    console.log('\n' + '='.repeat(60));
    console.log('Commodities Backfill Complete!');
    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
