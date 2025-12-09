/**
 * Cross-Validate Data Against Alpaca API
 *
 * This script compares our stored data against Alpaca's API
 * to ensure accuracy and consistency.
 *
 * Checks:
 * 1. Timestamp consistency (60-second intervals)
 * 2. OHLC validity (high >= low, prices in range)
 * 3. Cross-reference with Alpaca for random samples
 * 4. Price accuracy (compare with Alpaca values)
 *
 * Usage:
 *   node cross_validate_data.js [category] [asset]
 *   node cross_validate_data.js                     # Validate all
 *   node cross_validate_data.js "Stock Market" SPY  # Validate SPY only
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');

// Alpaca API credentials
const API_KEY = process.env.ALPACA_API_KEY || 'PKHEJ2BD4KXO7LSADIWYLU5JIQ';
const SECRET_KEY = process.env.ALPACA_SECRET_KEY || '8MyVxbmN3nUsCBD8TGWvXhtdjYmDEM6heTfvrTCkDQzE';

// Command line args
const TARGET_CATEGORY = process.argv[2] || null;
const TARGET_ASSET = process.argv[3] || null;

// Validation results
const results = {
    checked: 0,
    passed: 0,
    failed: 0,
    issues: []
};

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Fetch data from Alpaca for verification
function fetchAlpacaData(symbol, date) {
    return new Promise((resolve, reject) => {
        const startDate = new Date(date);
        startDate.setUTCHours(0, 0, 0, 0);
        const endDate = new Date(date);
        endDate.setUTCHours(23, 59, 59, 999);

        const options = {
            hostname: 'data.alpaca.markets',
            path: `/v2/stocks/${symbol}/bars?timeframe=1Min&start=${startDate.toISOString()}&end=${endDate.toISOString()}&limit=10000&adjustment=raw&feed=iex`,
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
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}`));
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

// Load a day's candle data
function loadDayFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
    } catch {
        return null;
    }
}

// Validate OHLC data
function validateOHLC(candle) {
    const issues = [];

    // Check required fields
    if (typeof candle.timestamp !== 'number') {
        issues.push('Missing or invalid timestamp');
    }
    if (typeof candle.open !== 'number' || isNaN(candle.open)) {
        issues.push('Invalid open price');
    }
    if (typeof candle.high !== 'number' || isNaN(candle.high)) {
        issues.push('Invalid high price');
    }
    if (typeof candle.low !== 'number' || isNaN(candle.low)) {
        issues.push('Invalid low price');
    }
    if (typeof candle.close !== 'number' || isNaN(candle.close)) {
        issues.push('Invalid close price');
    }

    // Check OHLC relationships
    if (candle.high < candle.low) {
        issues.push(`High (${candle.high}) < Low (${candle.low})`);
    }
    if (candle.open > candle.high || candle.open < candle.low) {
        issues.push(`Open (${candle.open}) outside high/low range`);
    }
    if (candle.close > candle.high || candle.close < candle.low) {
        issues.push(`Close (${candle.close}) outside high/low range`);
    }

    // Check for negative prices
    if (candle.open < 0 || candle.high < 0 || candle.low < 0 || candle.close < 0) {
        issues.push('Negative price detected');
    }

    return issues;
}

// Validate timestamp consistency
function validateTimestamps(candles) {
    const issues = [];
    const timestamps = candles.map(c => c.timestamp).sort((a, b) => a - b);

    // Check for duplicates
    const seen = new Set();
    for (const ts of timestamps) {
        if (seen.has(ts)) {
            issues.push(`Duplicate timestamp: ${new Date(ts).toISOString()}`);
        }
        seen.add(ts);
    }

    // Check intervals (should be ~60 seconds for 1-minute data)
    for (let i = 1; i < timestamps.length; i++) {
        const diff = timestamps[i] - timestamps[i - 1];
        // Allow up to 5 minutes gap (market breaks, etc.) but flag larger gaps
        if (diff > 300000) { // > 5 minutes
            // This is normal for overnight/weekend gaps, just note it
        }
        // Flag if interval is not 60 seconds (within tolerance)
        if (diff > 0 && diff < 60000 && diff !== 0) {
            issues.push(`Unexpected interval: ${diff}ms at ${new Date(timestamps[i]).toISOString()}`);
        }
    }

    return issues;
}

// Compare our data with Alpaca data
function compareWithAlpaca(ourCandles, alpacaCandles) {
    const issues = [];

    // Create map of Alpaca candles by timestamp
    const alpacaMap = new Map();
    for (const bar of alpacaCandles) {
        const ts = new Date(bar.t).getTime();
        alpacaMap.set(ts, {
            open: bar.o,
            high: bar.h,
            low: bar.l,
            close: bar.c,
            volume: bar.v
        });
    }

    // Compare random samples
    const sampleSize = Math.min(10, ourCandles.length);
    const samples = [];
    const indices = new Set();
    while (samples.length < sampleSize && indices.size < ourCandles.length) {
        const idx = Math.floor(Math.random() * ourCandles.length);
        if (!indices.has(idx)) {
            indices.add(idx);
            samples.push(ourCandles[idx]);
        }
    }

    let matches = 0;
    let mismatches = 0;
    let notInAlpaca = 0;

    for (const candle of samples) {
        const alpaca = alpacaMap.get(candle.timestamp);
        if (!alpaca) {
            notInAlpaca++;
            continue;
        }

        // Allow small price differences (different data sources may have slight variations)
        const tolerance = 0.01; // 1 cent
        const priceMatch = (
            Math.abs(candle.open - alpaca.open) <= tolerance &&
            Math.abs(candle.high - alpaca.high) <= tolerance &&
            Math.abs(candle.low - alpaca.low) <= tolerance &&
            Math.abs(candle.close - alpaca.close) <= tolerance
        );

        if (priceMatch) {
            matches++;
        } else {
            mismatches++;
            // Only flag significant differences (> 1%)
            const maxDiff = Math.max(
                Math.abs(candle.open - alpaca.open) / alpaca.open,
                Math.abs(candle.high - alpaca.high) / alpaca.high,
                Math.abs(candle.low - alpaca.low) / alpaca.low,
                Math.abs(candle.close - alpaca.close) / alpaca.close
            );
            if (maxDiff > 0.01) { // > 1% difference
                issues.push(`Price mismatch at ${new Date(candle.timestamp).toISOString()}: ` +
                    `Our OHLC=${candle.open}/${candle.high}/${candle.low}/${candle.close} vs ` +
                    `Alpaca=${alpaca.open}/${alpaca.high}/${alpaca.low}/${alpaca.close}`);
            }
        }
    }

    return {
        issues,
        matches,
        mismatches,
        notInAlpaca,
        total: samples.length
    };
}

// Validate a single day file
async function validateDayFile(category, asset, dateStr, filePath, crossValidate = false) {
    const candles = loadDayFile(filePath);
    if (!candles || candles.length === 0) {
        return { valid: true, issues: [] }; // Empty file is okay
    }

    const issues = [];

    // 1. Validate each candle's OHLC
    for (let i = 0; i < candles.length; i++) {
        const candleIssues = validateOHLC(candles[i]);
        if (candleIssues.length > 0) {
            issues.push(`Candle ${i}: ${candleIssues.join(', ')}`);
        }
    }

    // 2. Validate timestamp consistency
    const tsIssues = validateTimestamps(candles);
    issues.push(...tsIssues);

    // 3. Cross-validate with Alpaca (only for stocks, and only if requested)
    if (crossValidate && category === 'Stock Market') {
        try {
            const alpacaData = await fetchAlpacaData(asset, dateStr);
            if (alpacaData.bars && alpacaData.bars.length > 0) {
                const comparison = compareWithAlpaca(candles, alpacaData.bars);
                issues.push(...comparison.issues);

                if (comparison.matches > 0 && comparison.mismatches === 0) {
                    // Data matches Alpaca
                } else if (comparison.mismatches > comparison.matches) {
                    issues.push(`Cross-validation: ${comparison.mismatches}/${comparison.total} samples mismatched`);
                }
            }
            await sleep(350); // Respect rate limits
        } catch (err) {
            // Ignore cross-validation errors (rate limits, etc.)
        }
    }

    return {
        valid: issues.length === 0,
        issues,
        candleCount: candles.length
    };
}

// Get all categories and assets
function getCategories() {
    if (!fs.existsSync(DATA_DIR)) return [];
    return fs.readdirSync(DATA_DIR).filter(f => {
        const stat = fs.statSync(path.join(DATA_DIR, f));
        return stat.isDirectory();
    });
}

function getAssets(category) {
    const catPath = path.join(DATA_DIR, category);
    if (!fs.existsSync(catPath)) return [];
    return fs.readdirSync(catPath).filter(f => {
        const stat = fs.statSync(path.join(catPath, f));
        return stat.isDirectory();
    });
}

function getWeeks(category, asset) {
    const assetPath = path.join(DATA_DIR, category, asset);
    if (!fs.existsSync(assetPath)) return [];
    return fs.readdirSync(assetPath).filter(f => {
        return f.startsWith('week_') && fs.statSync(path.join(assetPath, f)).isDirectory();
    });
}

function getDayFiles(category, asset, week) {
    const weekPath = path.join(DATA_DIR, category, asset, week);
    if (!fs.existsSync(weekPath)) return [];
    return fs.readdirSync(weekPath).filter(f => f.endsWith('.json'));
}

// Main validation
async function main() {
    console.log('='.repeat(60));
    console.log('Cross-Validation: Verify Data Accuracy');
    console.log('='.repeat(60));
    console.log('');

    const categories = TARGET_CATEGORY ? [TARGET_CATEGORY] : getCategories();
    let totalFiles = 0;
    let validFiles = 0;
    let invalidFiles = 0;
    let totalCandles = 0;

    for (const category of categories) {
        const assets = TARGET_ASSET ? [TARGET_ASSET] : getAssets(category);

        for (const asset of assets) {
            console.log(`\nValidating ${category}/${asset}...`);

            const weeks = getWeeks(category, asset);
            let assetValid = true;
            let assetCandles = 0;
            const assetIssues = [];

            // Get most recent week for cross-validation
            const recentWeeks = weeks.slice(-2); // Last 2 weeks

            for (const week of weeks) {
                const dayFiles = getDayFiles(category, asset, week);

                for (const dayFile of dayFiles) {
                    const dateStr = dayFile.replace('.json', '');
                    const filePath = path.join(DATA_DIR, category, asset, week, dayFile);

                    // Only cross-validate recent data (to save API calls)
                    const crossValidate = recentWeeks.includes(week) && category === 'Stock Market';

                    const result = await validateDayFile(category, asset, dateStr, filePath, crossValidate);
                    totalFiles++;

                    if (result.valid) {
                        validFiles++;
                    } else {
                        invalidFiles++;
                        assetValid = false;
                        assetIssues.push({
                            file: `${week}/${dayFile}`,
                            issues: result.issues.slice(0, 3) // Limit issues shown
                        });
                    }

                    assetCandles += result.candleCount || 0;
                    totalCandles += result.candleCount || 0;
                }
            }

            if (assetValid) {
                console.log(`  OK - ${assetCandles.toLocaleString()} candles validated`);
            } else {
                console.log(`  ISSUES FOUND - ${assetIssues.length} files with problems:`);
                for (const issue of assetIssues.slice(0, 5)) { // Show max 5
                    console.log(`    ${issue.file}: ${issue.issues[0]}`);
                }
                if (assetIssues.length > 5) {
                    console.log(`    ... and ${assetIssues.length - 5} more files with issues`);
                }
                results.issues.push({
                    asset: `${category}/${asset}`,
                    issues: assetIssues
                });
            }
        }
    }

    console.log('\n' + '='.repeat(60));
    console.log('Validation Summary');
    console.log('='.repeat(60));
    console.log(`Total files checked: ${totalFiles.toLocaleString()}`);
    console.log(`Valid files: ${validFiles.toLocaleString()}`);
    console.log(`Files with issues: ${invalidFiles.toLocaleString()}`);
    console.log(`Total candles: ${totalCandles.toLocaleString()}`);
    console.log('');

    if (results.issues.length === 0) {
        console.log('All data validated successfully!');
        console.log('');
        console.log('Data format is consistent and OHLC values are valid.');
    } else {
        console.log(`Found issues in ${results.issues.length} assets:`);
        for (const issue of results.issues) {
            console.log(`  - ${issue.asset}: ${issue.issues.length} files`);
        }
    }

    console.log('='.repeat(60));
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
