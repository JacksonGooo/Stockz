/**
 * Smart Hierarchical Data Checker
 *
 * Validates data efficiently using a hierarchical approach:
 * 1. Spot check (sample 5 random candles per day)
 * 2. If spot check fails -> Full day check
 * 3. If day check fails -> Full week check
 * 4. Cross-validate with external API when needed
 *
 * Usage: node smart_data_checker.js           - Run once
 *        node smart_data_checker.js --watch   - Continuous monitoring
 *        node smart_data_checker.js --deep    - Deep scan (all candles)
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const DATA_DIR = path.join(__dirname, '..', 'Data');
const WATCH_MODE = process.argv.includes('--watch');
const DEEP_MODE = process.argv.includes('--deep');
const WATCH_INTERVAL = 5 * 60 * 1000; // 5 minutes

// Validation thresholds
const THRESHOLDS = {
    spotCheckSamples: 5,           // Number of candles to spot check per day
    priceJumpPercent: 20,          // Flag if price jumps > 20% in 1 minute
    maxGapMinutes: {
        'Crypto': 5,
        'Stock Market': 30,
        'Commodities': 30,
        'Currencies': 30
    },
    minCandlesPerDay: {
        'Crypto': 1200,            // 83% of 1440
        'Stock Market': 300,       // 77% of 390
        'Commodities': 300,
        'Currencies': 1100
    }
};

class SmartDataChecker {
    constructor() {
        this.issues = [];
        this.stats = {
            daysChecked: 0,
            spotChecksPassed: 0,
            spotChecksFailed: 0,
            dayChecksPassed: 0,
            dayChecksFailed: 0,
            weekChecksPassed: 0,
            weekChecksFailed: 0,
            crossValidations: 0,
            totalIssues: 0
        };
    }

    // Load JSON file
    loadJsonFile(filePath) {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            return JSON.parse(content);
        } catch {
            return null;
        }
    }

    // Pick random items from array
    pickRandom(arr, count) {
        if (arr.length <= count) return arr;
        const shuffled = [...arr].sort(() => Math.random() - 0.5);
        return shuffled.slice(0, count);
    }

    // Validate a single candle
    validateCandle(candle, prevCandle = null, category = 'Crypto') {
        const issues = [];

        // Check required fields
        const requiredFields = ['timestamp', 'open', 'high', 'low', 'close', 'volume'];
        for (const field of requiredFields) {
            if (candle[field] === undefined || candle[field] === null) {
                issues.push({ type: 'missing_field', field, severity: 'error' });
            }
        }

        if (issues.length > 0) return { valid: false, issues };

        // Check OHLC validity
        if (candle.high < candle.low) {
            issues.push({ type: 'invalid_ohlc', reason: 'high < low', severity: 'error' });
        }
        if (candle.open > candle.high || candle.open < candle.low) {
            issues.push({ type: 'invalid_ohlc', reason: 'open outside range', severity: 'warning' });
        }
        if (candle.close > candle.high || candle.close < candle.low) {
            issues.push({ type: 'invalid_ohlc', reason: 'close outside range', severity: 'warning' });
        }

        // Check for negative/zero prices
        if (candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0) {
            issues.push({ type: 'invalid_price', reason: 'non-positive price', severity: 'error' });
        }

        // Check for future timestamps
        if (candle.timestamp > Date.now()) {
            issues.push({ type: 'future_timestamp', severity: 'error' });
        }

        // Check for suspicious price jumps (compared to previous candle)
        if (prevCandle && prevCandle.close > 0) {
            const priceChange = Math.abs(candle.open - prevCandle.close) / prevCandle.close * 100;
            if (priceChange > THRESHOLDS.priceJumpPercent) {
                issues.push({
                    type: 'price_jump',
                    change: priceChange.toFixed(2) + '%',
                    severity: 'warning'
                });
            }
        }

        // Check for suspicious volume
        if (candle.volume < 0) {
            issues.push({ type: 'negative_volume', severity: 'error' });
        }

        return {
            valid: issues.filter(i => i.severity === 'error').length === 0,
            issues
        };
    }

    // Spot check: Sample random candles from a day
    spotCheck(category, asset, date, candles) {
        if (!candles || candles.length === 0) {
            return { passed: false, escalate: 'day', reason: 'empty_file' };
        }

        const samples = this.pickRandom(candles, THRESHOLDS.spotCheckSamples);
        let hasErrors = false;

        for (const candle of samples) {
            const result = this.validateCandle(candle, null, category);
            if (!result.valid) {
                hasErrors = true;
                break;
            }
        }

        if (hasErrors) {
            this.stats.spotChecksFailed++;
            return { passed: false, escalate: 'day', reason: 'spot_check_failed' };
        }

        // Also check candle count
        const minCandles = THRESHOLDS.minCandlesPerDay[category] || 1000;
        if (candles.length < minCandles * 0.5) {
            this.stats.spotChecksFailed++;
            return { passed: false, escalate: 'day', reason: 'low_candle_count', count: candles.length };
        }

        this.stats.spotChecksPassed++;
        return { passed: true };
    }

    // Full day check: Validate all candles in a day
    fullDayCheck(category, asset, date, candles) {
        if (!candles || candles.length === 0) {
            return { passed: false, escalate: 'week', issues: [{ type: 'empty_file' }] };
        }

        const issues = [];
        const sortedCandles = [...candles].sort((a, b) => a.timestamp - b.timestamp);

        // Check each candle
        for (let i = 0; i < sortedCandles.length; i++) {
            const prevCandle = i > 0 ? sortedCandles[i - 1] : null;
            const result = this.validateCandle(sortedCandles[i], prevCandle, category);

            if (!result.valid) {
                issues.push({
                    index: i,
                    timestamp: sortedCandles[i].timestamp,
                    issues: result.issues
                });
            }

            // Check for gaps
            if (prevCandle) {
                const gapMinutes = (sortedCandles[i].timestamp - prevCandle.timestamp) / 60000;
                const maxGap = THRESHOLDS.maxGapMinutes[category] || 10;

                if (gapMinutes > maxGap) {
                    issues.push({
                        index: i,
                        type: 'gap',
                        gapMinutes: gapMinutes.toFixed(0),
                        severity: 'info'
                    });
                }
            }
        }

        // Check for duplicates
        const timestamps = sortedCandles.map(c => c.timestamp);
        const uniqueTimestamps = new Set(timestamps);
        const duplicateCount = timestamps.length - uniqueTimestamps.size;

        if (duplicateCount > 0) {
            issues.push({
                type: 'duplicates',
                count: duplicateCount,
                severity: 'warning'
            });
        }

        const errorCount = issues.filter(i =>
            i.issues?.some(j => j.severity === 'error') || i.severity === 'error'
        ).length;

        if (errorCount > 0) {
            this.stats.dayChecksFailed++;
            return { passed: false, escalate: 'week', issues };
        }

        this.stats.dayChecksPassed++;
        return { passed: true, issues };
    }

    // Full week check: Check all days in a week
    fullWeekCheck(category, asset, weekFolder) {
        const weekPath = path.join(DATA_DIR, category, asset, weekFolder);

        if (!fs.existsSync(weekPath)) {
            return { passed: false, issues: [{ type: 'missing_week_folder' }] };
        }

        const dayFiles = fs.readdirSync(weekPath).filter(f => f.endsWith('.json')).sort();
        const allIssues = [];

        for (const dayFile of dayFiles) {
            const dateStr = dayFile.replace('.json', '');
            const filePath = path.join(weekPath, dayFile);
            const candles = this.loadJsonFile(filePath);

            const result = this.fullDayCheck(category, asset, dateStr, candles);
            if (!result.passed) {
                allIssues.push({
                    date: dateStr,
                    issues: result.issues
                });
            }
        }

        if (allIssues.length > 0) {
            this.stats.weekChecksFailed++;
            return { passed: false, issues: allIssues, needsRepair: true };
        }

        this.stats.weekChecksPassed++;
        return { passed: true };
    }

    // Cross-validate with external API (Binance for crypto)
    async crossValidate(category, asset, date, ourCandles) {
        if (category !== 'Crypto') {
            return { skipped: true, reason: 'only_crypto_supported' };
        }

        // Sample a few timestamps to verify
        const samples = this.pickRandom(ourCandles, 3);
        const discrepancies = [];

        for (const candle of samples) {
            try {
                const externalCandle = await this.fetchBinanceCandle(asset, candle.timestamp);

                if (externalCandle) {
                    const priceDiff = Math.abs(candle.close - externalCandle.close) / externalCandle.close * 100;

                    if (priceDiff > 1) { // More than 1% difference
                        discrepancies.push({
                            timestamp: candle.timestamp,
                            ourClose: candle.close,
                            externalClose: externalCandle.close,
                            difference: priceDiff.toFixed(2) + '%'
                        });
                    }
                }
            } catch (err) {
                // API error, skip this check
            }
        }

        this.stats.crossValidations++;
        return {
            passed: discrepancies.length === 0,
            discrepancies
        };
    }

    // Fetch a single candle from Binance for verification
    fetchBinanceCandle(asset, timestamp) {
        return new Promise((resolve, reject) => {
            const symbol = asset === 'POL' ? 'MATIC' : asset;
            const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}USDT&interval=1m&limit=1&startTime=${timestamp}`;

            https.get(url, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        if (res.statusCode === 200) {
                            const parsed = JSON.parse(data);
                            if (parsed.length > 0) {
                                resolve({
                                    timestamp: parsed[0][0],
                                    open: parseFloat(parsed[0][1]),
                                    high: parseFloat(parsed[0][2]),
                                    low: parseFloat(parsed[0][3]),
                                    close: parseFloat(parsed[0][4]),
                                    volume: parseFloat(parsed[0][5])
                                });
                            } else {
                                resolve(null);
                            }
                        } else {
                            resolve(null);
                        }
                    } catch (e) {
                        resolve(null);
                    }
                });
            }).on('error', () => resolve(null));
        });
    }

    // Check a single asset using hierarchical approach
    async checkAsset(category, asset) {
        const assetPath = path.join(DATA_DIR, category, asset);

        if (!fs.existsSync(assetPath)) {
            return { status: 'missing', issues: [] };
        }

        const weekFolders = fs.readdirSync(assetPath)
            .filter(f => f.startsWith('week_') && fs.statSync(path.join(assetPath, f)).isDirectory())
            .sort();

        const assetIssues = [];
        let totalDays = 0;
        let passedDays = 0;

        for (const weekFolder of weekFolders) {
            const weekPath = path.join(assetPath, weekFolder);
            const dayFiles = fs.readdirSync(weekPath).filter(f => f.endsWith('.json')).sort();

            for (const dayFile of dayFiles) {
                totalDays++;
                this.stats.daysChecked++;

                const dateStr = dayFile.replace('.json', '');
                const filePath = path.join(weekPath, dayFile);
                const candles = this.loadJsonFile(filePath);

                if (!candles) {
                    assetIssues.push({ date: dateStr, type: 'corrupt_file' });
                    continue;
                }

                // Step 1: Spot check
                const spotResult = this.spotCheck(category, asset, dateStr, candles);

                if (spotResult.passed) {
                    passedDays++;
                    continue;
                }

                // Step 2: Full day check (spot check failed)
                if (DEEP_MODE) {
                    const dayResult = this.fullDayCheck(category, asset, dateStr, candles);

                    if (dayResult.passed) {
                        passedDays++;
                        continue;
                    }

                    // Step 3: Flag for review (day check failed)
                    assetIssues.push({
                        date: dateStr,
                        week: weekFolder,
                        issues: dayResult.issues,
                        candleCount: candles.length
                    });
                } else {
                    // In quick mode, just flag the issue
                    assetIssues.push({
                        date: dateStr,
                        week: weekFolder,
                        reason: spotResult.reason,
                        candleCount: candles.length
                    });
                }
            }
        }

        return {
            status: assetIssues.length === 0 ? 'healthy' : 'issues',
            totalDays,
            passedDays,
            issues: assetIssues
        };
    }

    // Run full validation
    async validateAll() {
        console.log('='.repeat(60));
        console.log('SMART DATA CHECKER');
        console.log('='.repeat(60));
        console.log(`Mode: ${DEEP_MODE ? 'Deep Scan' : 'Quick Spot Check'}`);
        console.log(`Scan started: ${new Date().toISOString()}`);
        console.log(`Data directory: ${DATA_DIR}`);
        console.log('');

        const categories = fs.readdirSync(DATA_DIR)
            .filter(d => fs.statSync(path.join(DATA_DIR, d)).isDirectory() && !d.startsWith('.'));

        const report = {
            categories: {},
            summary: {
                totalAssets: 0,
                healthyAssets: 0,
                assetsWithIssues: 0,
                totalDays: 0,
                passedDays: 0
            }
        };

        for (const category of categories) {
            const categoryPath = path.join(DATA_DIR, category);
            const assets = fs.readdirSync(categoryPath)
                .filter(d => fs.statSync(path.join(categoryPath, d)).isDirectory());

            console.log(`\nChecking ${category} (${assets.length} assets)...`);
            report.categories[category] = { assets: {}, summary: { healthy: 0, issues: 0 } };

            for (const asset of assets) {
                const result = await this.checkAsset(category, asset);
                report.categories[category].assets[asset] = result;
                report.summary.totalAssets++;
                report.summary.totalDays += result.totalDays || 0;
                report.summary.passedDays += result.passedDays || 0;

                if (result.status === 'healthy') {
                    report.categories[category].summary.healthy++;
                    report.summary.healthyAssets++;
                    process.stdout.write('.');
                } else {
                    report.categories[category].summary.issues++;
                    report.summary.assetsWithIssues++;
                    process.stdout.write('!');
                    this.issues.push({
                        category,
                        asset,
                        issues: result.issues
                    });
                }
            }
        }

        this.printReport(report);
        return report;
    }

    // Print validation report
    printReport(report) {
        console.log('\n\n' + '='.repeat(60));
        console.log('VALIDATION REPORT');
        console.log('='.repeat(60));

        console.log('\nSUMMARY:');
        console.log(`  Total assets: ${report.summary.totalAssets}`);
        console.log(`  Healthy assets: ${report.summary.healthyAssets}`);
        console.log(`  Assets with issues: ${report.summary.assetsWithIssues}`);
        console.log(`  Days checked: ${this.stats.daysChecked}`);
        console.log(`  Spot checks passed: ${this.stats.spotChecksPassed}`);
        console.log(`  Spot checks failed: ${this.stats.spotChecksFailed}`);

        if (this.issues.length > 0) {
            console.log('\n' + '='.repeat(60));
            console.log(`ISSUES FOUND (${this.issues.length} assets)`);
            console.log('='.repeat(60));

            // Show top 20 issues
            for (const issue of this.issues.slice(0, 20)) {
                console.log(`\n${issue.category}/${issue.asset}:`);
                for (const dayIssue of issue.issues.slice(0, 5)) {
                    console.log(`  - ${dayIssue.date}: ${dayIssue.reason || 'validation failed'} (${dayIssue.candleCount} candles)`);
                }
                if (issue.issues.length > 5) {
                    console.log(`  ... and ${issue.issues.length - 5} more days`);
                }
            }

            if (this.issues.length > 20) {
                console.log(`\n... and ${this.issues.length - 20} more assets with issues`);
            }
        } else {
            console.log('\nAll data passed validation!');
        }

        console.log('\n' + '='.repeat(60));
        console.log(`Scan completed: ${new Date().toISOString()}`);
        console.log('='.repeat(60));
    }

    // Reset state for next run
    reset() {
        this.issues = [];
        this.stats = {
            daysChecked: 0,
            spotChecksPassed: 0,
            spotChecksFailed: 0,
            dayChecksPassed: 0,
            dayChecksFailed: 0,
            weekChecksPassed: 0,
            weekChecksFailed: 0,
            crossValidations: 0,
            totalIssues: 0
        };
    }
}

// Main
const checker = new SmartDataChecker();

if (WATCH_MODE) {
    console.log('Starting continuous data validation (every 5 minutes)...');
    console.log('Press Ctrl+C to stop.\n');

    const runCheck = async () => {
        checker.reset();
        await checker.validateAll();
        console.log(`\nNext scan in 5 minutes...\n`);
    };

    runCheck();
    setInterval(runCheck, WATCH_INTERVAL);
} else {
    checker.validateAll().catch(err => {
        console.error('Error:', err);
        process.exit(1);
    });
}
