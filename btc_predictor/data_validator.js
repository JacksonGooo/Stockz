/**
 * Data Validator & Monitor
 *
 * Continuously scans all data for issues:
 * - Missing days
 * - Suspicious gaps (> 5 minute gaps in crypto)
 * - Incorrect timestamps
 * - Duplicate candles
 * - Candle count anomalies
 * - Future timestamps (data corruption)
 *
 * Usage: node data_validator.js           - Run once and report
 *        node data_validator.js --watch   - Continuous monitoring every 5 minutes
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');
const WATCH_MODE = process.argv.includes('--watch');
const WATCH_INTERVAL = 5 * 60 * 1000; // 5 minutes

// Expected candles per day for each market type
const EXPECTED_CANDLES = {
    'Crypto': 1440,        // 24h * 60min
    'Stock Market': 390,   // 6.5h trading day
    'Commodities': 1320,   // ~22 hours
    'Currencies': 1320,    // ~22 hours (forex)
};

// Tolerance thresholds
const THRESHOLDS = {
    missingDayWarning: 0.5,    // Warn if < 50% of expected candles
    gapWarningMinutes: 10,      // Warn on gaps > 10 minutes
    duplicateThreshold: 0.01,   // Warn if > 1% duplicates
};

class DataValidator {
    constructor() {
        this.issues = [];
        this.summary = {
            totalAssets: 0,
            assetsWithIssues: 0,
            totalCandles: 0,
            missingDays: 0,
            largeGaps: 0,
            duplicates: 0,
            futureTimestamps: 0,
            corruptFiles: 0,
        };
    }

    addIssue(severity, category, asset, message, details = {}) {
        this.issues.push({
            severity, // 'error', 'warning', 'info'
            category,
            asset,
            message,
            details,
            timestamp: new Date().toISOString(),
        });
    }

    loadJsonFile(filePath) {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            return JSON.parse(content);
        } catch (err) {
            return null;
        }
    }

    validateCandle(candle, category, asset, dateStr) {
        const issues = [];
        const now = Date.now();

        // Check for future timestamps
        if (candle.timestamp > now) {
            issues.push({ type: 'future_timestamp', timestamp: candle.timestamp });
            this.summary.futureTimestamps++;
        }

        // Check for missing OHLCV fields
        const requiredFields = ['timestamp', 'open', 'high', 'low', 'close', 'volume'];
        for (const field of requiredFields) {
            if (candle[field] === undefined || candle[field] === null) {
                issues.push({ type: 'missing_field', field });
            }
        }

        // Check for impossible OHLC values
        if (candle.high < candle.low) {
            issues.push({ type: 'invalid_ohlc', reason: 'high < low' });
        }
        if (candle.open > candle.high || candle.open < candle.low) {
            issues.push({ type: 'invalid_ohlc', reason: 'open outside range' });
        }
        if (candle.close > candle.high || candle.close < candle.low) {
            issues.push({ type: 'invalid_ohlc', reason: 'close outside range' });
        }

        // Check for negative values
        if (candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0) {
            issues.push({ type: 'negative_price' });
        }

        return issues;
    }

    validateDayFile(filePath, category, asset, dateStr) {
        const candles = this.loadJsonFile(filePath);

        if (candles === null) {
            this.addIssue('error', category, asset, `Corrupt JSON file: ${dateStr}`, { file: filePath });
            this.summary.corruptFiles++;
            return;
        }

        if (!Array.isArray(candles)) {
            this.addIssue('error', category, asset, `Invalid data format: ${dateStr}`, { file: filePath });
            this.summary.corruptFiles++;
            return;
        }

        if (candles.length === 0) {
            this.addIssue('warning', category, asset, `Empty file: ${dateStr}`, { file: filePath });
            return;
        }

        // Check candle count
        const expected = EXPECTED_CANDLES[category] || 1440;
        const completeness = candles.length / expected;
        if (completeness < THRESHOLDS.missingDayWarning) {
            this.addIssue('warning', category, asset,
                `Low candle count: ${candles.length}/${expected} (${(completeness*100).toFixed(1)}%) on ${dateStr}`,
                { expected, actual: candles.length, date: dateStr }
            );
            this.summary.missingDays++;
        }

        // Check for duplicates
        const timestamps = candles.map(c => c.timestamp);
        const uniqueTimestamps = new Set(timestamps);
        const duplicateCount = timestamps.length - uniqueTimestamps.size;
        if (duplicateCount > timestamps.length * THRESHOLDS.duplicateThreshold) {
            this.addIssue('warning', category, asset,
                `${duplicateCount} duplicate timestamps on ${dateStr}`,
                { duplicates: duplicateCount, date: dateStr }
            );
            this.summary.duplicates += duplicateCount;
        }

        // Sort candles and check for gaps
        const sortedCandles = [...candles].sort((a, b) => a.timestamp - b.timestamp);
        for (let i = 1; i < sortedCandles.length; i++) {
            const gap = (sortedCandles[i].timestamp - sortedCandles[i-1].timestamp) / 60000; // minutes
            const maxGap = category === 'Crypto' ? THRESHOLDS.gapWarningMinutes : 60; // More tolerant for stocks

            if (gap > maxGap) {
                this.addIssue('info', category, asset,
                    `${gap.toFixed(0)} minute gap on ${dateStr}`,
                    { gapMinutes: gap, date: dateStr }
                );
                this.summary.largeGaps++;
            }
        }

        // Validate each candle
        let candleIssues = 0;
        for (const candle of candles) {
            const issues = this.validateCandle(candle, category, asset, dateStr);
            candleIssues += issues.length;
        }

        if (candleIssues > 0) {
            this.addIssue('warning', category, asset,
                `${candleIssues} candle validation issues on ${dateStr}`,
                { issues: candleIssues, date: dateStr }
            );
        }

        this.summary.totalCandles += candles.length;
    }

    validateAsset(category, asset) {
        const assetPath = path.join(DATA_DIR, category, asset);

        if (!fs.existsSync(assetPath)) {
            return;
        }

        const weekFolders = fs.readdirSync(assetPath)
            .filter(f => f.startsWith('week_') && fs.statSync(path.join(assetPath, f)).isDirectory());

        for (const weekFolder of weekFolders) {
            const weekPath = path.join(assetPath, weekFolder);
            const dayFiles = fs.readdirSync(weekPath)
                .filter(f => f.endsWith('.json'));

            for (const dayFile of dayFiles) {
                const dateStr = dayFile.replace('.json', '');
                const filePath = path.join(weekPath, dayFile);
                this.validateDayFile(filePath, category, asset, dateStr);
            }
        }
    }

    validateAll() {
        console.log('='.repeat(60));
        console.log('DATA VALIDATION REPORT');
        console.log('='.repeat(60));
        console.log(`Scan started: ${new Date().toISOString()}`);
        console.log(`Data directory: ${DATA_DIR}`);
        console.log('');

        const categories = fs.readdirSync(DATA_DIR)
            .filter(d => fs.statSync(path.join(DATA_DIR, d)).isDirectory() && !d.startsWith('.'));

        for (const category of categories) {
            const categoryPath = path.join(DATA_DIR, category);
            const assets = fs.readdirSync(categoryPath)
                .filter(d => fs.statSync(path.join(categoryPath, d)).isDirectory());

            console.log(`Validating ${category} (${assets.length} assets)...`);

            for (const asset of assets) {
                this.summary.totalAssets++;
                const issuesBefore = this.issues.length;
                this.validateAsset(category, asset);
                if (this.issues.length > issuesBefore) {
                    this.summary.assetsWithIssues++;
                }
            }
        }

        this.printReport();
    }

    printReport() {
        console.log('');
        console.log('='.repeat(60));
        console.log('SUMMARY');
        console.log('='.repeat(60));
        console.log(`Total assets scanned: ${this.summary.totalAssets}`);
        console.log(`Assets with issues: ${this.summary.assetsWithIssues}`);
        console.log(`Total candles: ${this.summary.totalCandles.toLocaleString()}`);
        console.log('');
        console.log('Issues found:');
        console.log(`  - Missing/low candle days: ${this.summary.missingDays}`);
        console.log(`  - Large gaps: ${this.summary.largeGaps}`);
        console.log(`  - Duplicate timestamps: ${this.summary.duplicates}`);
        console.log(`  - Future timestamps: ${this.summary.futureTimestamps}`);
        console.log(`  - Corrupt files: ${this.summary.corruptFiles}`);
        console.log('');

        // Print errors first, then warnings
        const errors = this.issues.filter(i => i.severity === 'error');
        const warnings = this.issues.filter(i => i.severity === 'warning').slice(0, 20); // Limit warnings

        if (errors.length > 0) {
            console.log('='.repeat(60));
            console.log(`ERRORS (${errors.length})`);
            console.log('='.repeat(60));
            for (const issue of errors) {
                console.log(`  [ERROR] ${issue.category}/${issue.asset}: ${issue.message}`);
            }
        }

        if (warnings.length > 0) {
            console.log('');
            console.log('='.repeat(60));
            console.log(`WARNINGS (showing ${warnings.length} of ${this.issues.filter(i => i.severity === 'warning').length})`);
            console.log('='.repeat(60));
            for (const issue of warnings) {
                console.log(`  [WARN] ${issue.category}/${issue.asset}: ${issue.message}`);
            }
        }

        if (errors.length === 0 && warnings.length === 0) {
            console.log('✅ No critical issues found!');
        }

        console.log('');
        console.log(`Scan completed: ${new Date().toISOString()}`);
        console.log('='.repeat(60));
    }

    reset() {
        this.issues = [];
        this.summary = {
            totalAssets: 0,
            assetsWithIssues: 0,
            totalCandles: 0,
            missingDays: 0,
            largeGaps: 0,
            duplicates: 0,
            futureTimestamps: 0,
            corruptFiles: 0,
        };
    }
}

// Main
const validator = new DataValidator();

if (WATCH_MODE) {
    console.log('Starting continuous data validation (every 5 minutes)...');
    console.log('Press Ctrl+C to stop.\n');

    const runValidation = () => {
        validator.reset();
        validator.validateAll();
        console.log(`\nNext scan in 5 minutes...\n`);
    };

    runValidation();
    setInterval(runValidation, WATCH_INTERVAL);
} else {
    validator.validateAll();
}
