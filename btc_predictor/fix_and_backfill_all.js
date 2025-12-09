/**
 * OVERNIGHT DATA FIX & BACKFILL SCRIPT
 *
 * This script will:
 * 1. Reorganize all files into correct week folders
 * 2. Merge duplicate files
 * 3. Verify data completeness (every minute)
 * 4. Run Polygon backfill for ALL assets (2 years)
 *
 * Run this overnight - it will take many hours!
 *
 * Usage:
 *   node fix_and_backfill_all.js
 */

const fs = require('fs');
const path = require('path');
const { spawn, execSync } = require('child_process');

const DATA_DIR = path.join(__dirname, '..', 'Data');
const LOG_FILE = path.join(__dirname, 'overnight_log.txt');

// Expected minutes per day
const EXPECTED_MINUTES = {
    'Crypto': 1440,           // 24 hours
    'Stock Market': 390,      // 6.5 hours (9:30-4:00 ET)
    'Commodities': 1320,      // ~22 hours
    'Currencies': 1320,       // ~22 hours
};

// All assets to backfill from Polygon (EXPANDED - matches polygon_backfill.js)
const POLYGON_ASSETS = {
    'Crypto': [
        // Major
        'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT',
        'POL', 'LINK', 'LTC', 'UNI', 'ATOM', 'XLM', 'ALGO',
        // DeFi
        'BCH', 'ETC', 'FIL', 'AAVE', 'MKR', 'COMP', 'SNX', 'SUSHI',
        'YFI', 'CRV', 'BAL',
        // Layer 2 & New
        'SHIB', 'APE', 'ARB', 'OP', 'NEAR', 'FTM',
        // Metaverse & Gaming
        'SAND', 'MANA', 'AXS', 'ENJ', 'GRT', 'LRC',
        // More Altcoins
        'ICP', 'VET', 'HBAR', 'QNT', 'EGLD', 'THETA', 'XTZ', 'EOS',
        'FLOW', 'CHZ', 'KAVA', 'ROSE', 'ZEC', 'DASH', 'NEO', 'WAVES',
        'ZIL', 'ENS', 'DYDX', 'ONE', 'CELO', 'ANKR', 'OMG', 'SKL',
        'NMR', 'OGN', 'BAND'
    ],
    'Stock Market': [
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
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'MPC', 'VLO', 'PSX',
        // Tech & Internet
        'NFLX', 'PYPL', 'SQ', 'COIN', 'UBER', 'ABNB', 'LYFT', 'DASH', 'SHOP', 'EBAY', 'ETSY', 'ZM', 'ROKU', 'SPOT', 'SNAP', 'PINS', 'TTD',
        // Industrial
        'CAT', 'BA', 'UPS', 'FDX', 'HON', 'GE', 'RTX', 'LMT', 'NOC', 'GD', 'DE', 'MMM',
        // Telecom & Media
        'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'WBD', 'PARA',
        // Auto
        'GM', 'F', 'RIVN', 'LCID',
        // Meme Stocks
        'GME', 'AMC', 'SOFI', 'HOOD', 'AFRM'
    ],
    'Commodities': [
        'GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM',
        'OIL', 'NATGAS', 'COPPER',
        'CORN', 'WHEAT', 'SOYBEAN',
        'DBC', 'GSG', 'PDBC'
    ],
    'Currencies': [
        // Major pairs
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
        // Cross pairs
        'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'EURAUD', 'EURCHF',
        'GBPCHF', 'GBPAUD', 'AUDNZD', 'NZDJPY', 'CHFJPY', 'EURCAD', 'AUDCAD', 'GBPCAD',
        // Emerging markets
        'USDMXN', 'USDZAR', 'USDTRY', 'USDBRL', 'USDINR', 'USDCNY', 'USDSGD',
        'USDHKD', 'USDKRW', 'USDSEK', 'USDNOK', 'USDDKK', 'USDPLN', 'USDHUF',
        'USDCZK', 'USDTHB', 'USDMYR', 'USDIDR', 'USDPHP'
    ],
};

// ============================================================
// Logging
// ============================================================
function log(message) {
    const timestamp = new Date().toISOString();
    const logLine = `[${timestamp}] ${message}`;
    console.log(logLine);
    fs.appendFileSync(LOG_FILE, logLine + '\n');
}

// ============================================================
// Week Calculation (FIXED)
// ============================================================
function getCorrectWeekStart(dateStr) {
    // dateStr is like "2025-12-06"
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(year, month - 1, day);
    const dayOfWeek = date.getDay(); // 0 = Sunday, 1 = Monday, etc.

    // Week starts on Monday
    // If Sunday (0), go back 6 days
    // If Monday (1), stay
    // If Tuesday (2), go back 1 day, etc.
    const daysToSubtract = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - daysToSubtract);

    return weekStart.toISOString().split('T')[0];
}

// ============================================================
// Step 1: Reorganize Files Into Correct Week Folders
// ============================================================
function reorganizeFiles() {
    log('='.repeat(60));
    log('STEP 1: REORGANIZING FILES INTO CORRECT WEEK FOLDERS');
    log('='.repeat(60));

    let filesMoved = 0;
    let filesFixed = 0;
    let foldersDeleted = 0;

    const categories = fs.readdirSync(DATA_DIR)
        .filter(d => {
            const p = path.join(DATA_DIR, d);
            return fs.statSync(p).isDirectory() && !d.startsWith('.');
        });

    for (const category of categories) {
        const categoryPath = path.join(DATA_DIR, category);
        const assets = fs.readdirSync(categoryPath)
            .filter(d => fs.statSync(path.join(categoryPath, d)).isDirectory());

        for (const asset of assets) {
            const assetPath = path.join(categoryPath, asset);
            log(`Processing ${category}/${asset}...`);

            // Collect all day files from all week folders
            const allDayFiles = new Map(); // date -> {candles, sources}

            const weekDirs = fs.readdirSync(assetPath)
                .filter(d => d.startsWith('week_'));

            for (const weekDir of weekDirs) {
                const weekPath = path.join(assetPath, weekDir);
                const files = fs.readdirSync(weekPath).filter(f => f.endsWith('.json'));

                for (const file of files) {
                    const dateStr = file.replace('.json', '');
                    const filePath = path.join(weekPath, file);

                    try {
                        const candles = JSON.parse(fs.readFileSync(filePath, 'utf8'));

                        if (!allDayFiles.has(dateStr)) {
                            allDayFiles.set(dateStr, { candles: [], sources: [] });
                        }

                        const existing = allDayFiles.get(dateStr);
                        existing.candles.push(...candles);
                        existing.sources.push(filePath);
                    } catch (e) {
                        log(`  Error reading ${filePath}: ${e.message}`);
                    }
                }
            }

            // Delete all old week folders
            for (const weekDir of weekDirs) {
                const weekPath = path.join(assetPath, weekDir);
                fs.rmSync(weekPath, { recursive: true, force: true });
                foldersDeleted++;
            }

            // Recreate with correct organization
            for (const [dateStr, data] of allDayFiles) {
                const correctWeekStart = getCorrectWeekStart(dateStr);
                const weekFolder = path.join(assetPath, `week_${correctWeekStart}`);

                if (!fs.existsSync(weekFolder)) {
                    fs.mkdirSync(weekFolder, { recursive: true });
                }

                // Merge and deduplicate candles
                const uniqueCandles = new Map();
                for (const candle of data.candles) {
                    uniqueCandles.set(candle.timestamp, candle);
                }

                const sortedCandles = Array.from(uniqueCandles.values())
                    .sort((a, b) => a.timestamp - b.timestamp);

                const dayFile = path.join(weekFolder, `${dateStr}.json`);
                fs.writeFileSync(dayFile, JSON.stringify(sortedCandles, null, 2));

                if (data.sources.length > 1) {
                    log(`  Merged ${data.sources.length} files for ${dateStr}: ${sortedCandles.length} candles`);
                    filesFixed++;
                }
                filesMoved++;
            }
        }
    }

    log(`Reorganization complete: ${filesMoved} files, ${filesFixed} merged, ${foldersDeleted} old folders deleted`);
    return { filesMoved, filesFixed, foldersDeleted };
}

// ============================================================
// Step 2: Verify Data Completeness
// ============================================================
function verifyDataCompleteness() {
    log('');
    log('='.repeat(60));
    log('STEP 2: VERIFYING DATA COMPLETENESS');
    log('='.repeat(60));

    const issues = [];

    const categories = fs.readdirSync(DATA_DIR)
        .filter(d => {
            const p = path.join(DATA_DIR, d);
            return fs.statSync(p).isDirectory() && !d.startsWith('.');
        });

    for (const category of categories) {
        const categoryPath = path.join(DATA_DIR, category);
        const expectedMinutes = EXPECTED_MINUTES[category] || 1440;

        const assets = fs.readdirSync(categoryPath)
            .filter(d => fs.statSync(path.join(categoryPath, d)).isDirectory());

        for (const asset of assets) {
            const assetPath = path.join(categoryPath, asset);
            let totalCandles = 0;
            let totalDays = 0;
            let completeDays = 0;
            let gaps = [];

            const weekDirs = fs.readdirSync(assetPath)
                .filter(d => d.startsWith('week_'))
                .sort();

            for (const weekDir of weekDirs) {
                const weekPath = path.join(assetPath, weekDir);
                const files = fs.readdirSync(weekPath).filter(f => f.endsWith('.json')).sort();

                for (const file of files) {
                    const dateStr = file.replace('.json', '');
                    const filePath = path.join(weekPath, file);

                    try {
                        const candles = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                        totalCandles += candles.length;
                        totalDays++;

                        const coverage = candles.length / expectedMinutes;
                        if (coverage >= 0.95) {
                            completeDays++;
                        } else if (coverage < 0.5) {
                            gaps.push({
                                date: dateStr,
                                candles: candles.length,
                                expected: expectedMinutes,
                                coverage: (coverage * 100).toFixed(1) + '%'
                            });
                        }

                        // Check for gaps within the day
                        if (candles.length > 1) {
                            candles.sort((a, b) => a.timestamp - b.timestamp);
                            for (let i = 1; i < candles.length; i++) {
                                const gap = (candles[i].timestamp - candles[i-1].timestamp) / 60000;
                                if (gap > 5) { // More than 5 minute gap
                                    gaps.push({
                                        date: dateStr,
                                        type: 'intra-day gap',
                                        gapMinutes: Math.round(gap),
                                        at: new Date(candles[i-1].timestamp).toISOString()
                                    });
                                }
                            }
                        }
                    } catch (e) {
                        // Skip
                    }
                }
            }

            const coverage = totalDays > 0 ? completeDays / totalDays : 0;

            if (coverage < 0.9 || gaps.length > 10) {
                issues.push({
                    asset: `${category}/${asset}`,
                    totalCandles,
                    totalDays,
                    completeDays,
                    coverage: (coverage * 100).toFixed(1) + '%',
                    significantGaps: gaps.slice(0, 5).length
                });
            }

            log(`  ${category}/${asset}: ${totalCandles.toLocaleString()} candles, ${completeDays}/${totalDays} complete (${(coverage*100).toFixed(1)}%)`);
        }
    }

    if (issues.length > 0) {
        log('');
        log('Assets with issues:');
        for (const issue of issues) {
            log(`  ${issue.asset}: ${issue.coverage} coverage, ${issue.significantGaps} gaps`);
        }
    }

    return issues;
}

// ============================================================
// Step 3: Run Polygon Backfill for All Assets
// ============================================================
async function runPolygonBackfill(category, asset, days) {
    return new Promise((resolve) => {
        log(`  Starting backfill for ${category}/${asset} (${days} days)...`);

        const script = path.join(__dirname, 'polygon_backfill.js');
        const proc = spawn('node', [script, category, asset, days.toString()], {
            cwd: __dirname,
            stdio: 'pipe'
        });

        let output = '';
        proc.stdout.on('data', (data) => {
            output += data.toString();
        });
        proc.stderr.on('data', (data) => {
            output += data.toString();
        });

        proc.on('close', (code) => {
            // Extract summary
            const match = output.match(/Total new candles saved: (\d+)/);
            const candlesSaved = match ? parseInt(match[1]) : 0;

            if (candlesSaved > 0) {
                log(`  ${category}/${asset}: +${candlesSaved.toLocaleString()} new candles`);
            } else if (code !== 0) {
                log(`  ${category}/${asset}: Error or no data available`);
            }

            resolve({ code, candlesSaved });
        });

        // Timeout after 2 hours per asset
        setTimeout(() => {
            proc.kill();
            resolve({ code: -1, candlesSaved: 0 });
        }, 2 * 60 * 60 * 1000);
    });
}

async function runAllBackfills() {
    log('');
    log('='.repeat(60));
    log('STEP 3: RUNNING POLYGON BACKFILL FOR ALL ASSETS');
    log('='.repeat(60));
    log('');
    log('This will take many hours. Go to sleep!');
    log('');

    const DAYS_TO_BACKFILL = 730; // 2 years
    let totalAssetsProcessed = 0;
    let totalNewCandles = 0;

    const startTime = Date.now();

    for (const [category, assets] of Object.entries(POLYGON_ASSETS)) {
        log(`\nCategory: ${category}`);
        log('-'.repeat(40));

        for (const asset of assets) {
            try {
                const result = await runPolygonBackfill(category, asset, DAYS_TO_BACKFILL);
                totalAssetsProcessed++;
                totalNewCandles += result.candlesSaved;

                const elapsed = (Date.now() - startTime) / 1000 / 60;
                const remaining = (elapsed / totalAssetsProcessed) * (getTotalAssetCount() - totalAssetsProcessed);
                log(`  Progress: ${totalAssetsProcessed}/${getTotalAssetCount()} | Elapsed: ${elapsed.toFixed(0)}m | ETA: ${remaining.toFixed(0)}m`);

            } catch (e) {
                log(`  ${category}/${asset}: Exception - ${e.message}`);
            }
        }
    }

    const totalElapsed = (Date.now() - startTime) / 1000 / 60 / 60;
    log('');
    log('='.repeat(60));
    log(`BACKFILL COMPLETE: ${totalAssetsProcessed} assets, ${totalNewCandles.toLocaleString()} new candles`);
    log(`Total time: ${totalElapsed.toFixed(1)} hours`);
    log('='.repeat(60));

    return { totalAssetsProcessed, totalNewCandles, hours: totalElapsed };
}

function getTotalAssetCount() {
    let count = 0;
    for (const assets of Object.values(POLYGON_ASSETS)) {
        count += assets.length;
    }
    return count;
}

// ============================================================
// Main
// ============================================================
async function main() {
    // Clear log file
    fs.writeFileSync(LOG_FILE, '');

    log('='.repeat(60));
    log('OVERNIGHT DATA FIX & BACKFILL SCRIPT');
    log('='.repeat(60));
    log(`Started at: ${new Date().toISOString()}`);
    log(`Data directory: ${DATA_DIR}`);
    log(`Log file: ${LOG_FILE}`);
    log('');

    const startTime = Date.now();

    // Step 1: Reorganize files
    const reorgResult = reorganizeFiles();

    // Step 2: Verify completeness
    const issues = verifyDataCompleteness();

    // Step 3: Run backfills
    const backfillResult = await runAllBackfills();

    // Final verification
    log('');
    log('='.repeat(60));
    log('FINAL VERIFICATION');
    log('='.repeat(60));
    verifyDataCompleteness();

    const totalElapsed = (Date.now() - startTime) / 1000 / 60 / 60;

    log('');
    log('='.repeat(60));
    log('OVERNIGHT SCRIPT COMPLETE');
    log('='.repeat(60));
    log(`Total time: ${totalElapsed.toFixed(1)} hours`);
    log(`Files reorganized: ${reorgResult.filesMoved}`);
    log(`Files merged: ${reorgResult.filesFixed}`);
    log(`New candles from Polygon: ${backfillResult.totalNewCandles.toLocaleString()}`);
    log(`Finished at: ${new Date().toISOString()}`);
    log('='.repeat(60));
}

main().catch(err => {
    log(`FATAL ERROR: ${err.message}`);
    console.error(err);
    process.exit(1);
});
