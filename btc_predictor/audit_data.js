/**
 * Data Quality Audit Script
 *
 * Checks all data files for:
 * - Missing minutes (gaps in data)
 * - Missing days
 * - Missing weeks
 * - Data completeness per asset
 * - Total candle counts
 *
 * Usage:
 *   node audit_data.js              - Audit all assets
 *   node audit_data.js Crypto       - Audit specific category
 *   node audit_data.js Crypto BTC   - Audit specific asset
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');

// Expected minutes per day for different asset types
const EXPECTED_MINUTES = {
    'Crypto': 1440,           // 24 hours = 1440 minutes
    'Stock Market': 390,      // 6.5 hours (9:30-4:00 ET) = 390 minutes
    'Commodities': 1260,      // ~21 hours for forex/commodities
    'Currencies': 1260,       // ~21 hours for forex
};

function formatNumber(num) {
    return num.toLocaleString();
}

function formatPercent(num) {
    return (num * 100).toFixed(1) + '%';
}

function loadDayFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
    } catch {
        return [];
    }
}

function analyzeDay(candles, expectedMinutes) {
    if (candles.length === 0) return { minutes: 0, gaps: [], coverage: 0 };

    // Sort by timestamp
    candles.sort((a, b) => a.timestamp - b.timestamp);

    const gaps = [];
    let totalGapMinutes = 0;

    for (let i = 1; i < candles.length; i++) {
        const diff = (candles[i].timestamp - candles[i-1].timestamp) / 60000;
        if (diff > 1.5) { // More than 1.5 minutes gap
            const gapMinutes = Math.round(diff - 1);
            gaps.push({
                from: new Date(candles[i-1].timestamp).toISOString(),
                to: new Date(candles[i].timestamp).toISOString(),
                minutes: gapMinutes
            });
            totalGapMinutes += gapMinutes;
        }
    }

    const coverage = Math.min(1, candles.length / expectedMinutes);

    return {
        minutes: candles.length,
        gaps,
        totalGapMinutes,
        coverage,
        firstCandle: new Date(candles[0].timestamp).toISOString(),
        lastCandle: new Date(candles[candles.length - 1].timestamp).toISOString(),
    };
}

function auditAsset(category, asset) {
    const assetDir = path.join(DATA_DIR, category, asset);

    if (!fs.existsSync(assetDir)) {
        return null;
    }

    const expectedMinutes = EXPECTED_MINUTES[category] || 1440;
    const weekDirs = fs.readdirSync(assetDir)
        .filter(d => d.startsWith('week_'))
        .sort();

    let totalCandles = 0;
    let totalDays = 0;
    let completeDays = 0;
    let partialDays = 0;
    let missingDays = 0;
    let totalGapMinutes = 0;
    const dayDetails = [];
    const weekSummary = [];

    // Track date range
    let firstDate = null;
    let lastDate = null;

    for (const weekDir of weekDirs) {
        const weekPath = path.join(assetDir, weekDir);
        const files = fs.readdirSync(weekPath)
            .filter(f => f.endsWith('.json'))
            .sort();

        let weekCandles = 0;
        let weekComplete = 0;

        for (const file of files) {
            const dateStr = file.replace('.json', '');
            const filePath = path.join(weekPath, file);
            const candles = loadDayFile(filePath);

            if (!firstDate || dateStr < firstDate) firstDate = dateStr;
            if (!lastDate || dateStr > lastDate) lastDate = dateStr;

            const analysis = analyzeDay(candles, expectedMinutes);
            totalCandles += analysis.minutes;
            weekCandles += analysis.minutes;
            totalDays++;

            if (analysis.coverage >= 0.95) {
                completeDays++;
                weekComplete++;
            } else if (analysis.minutes > 0) {
                partialDays++;
            } else {
                missingDays++;
            }

            totalGapMinutes += analysis.totalGapMinutes;

            if (analysis.gaps.length > 0 || analysis.coverage < 0.9) {
                dayDetails.push({
                    date: dateStr,
                    candles: analysis.minutes,
                    expected: expectedMinutes,
                    coverage: analysis.coverage,
                    gaps: analysis.gaps.length,
                    gapMinutes: analysis.totalGapMinutes,
                });
            }
        }

        weekSummary.push({
            week: weekDir,
            days: files.length,
            candles: weekCandles,
            complete: weekComplete,
        });
    }

    return {
        category,
        asset,
        totalCandles,
        totalDays,
        completeDays,
        partialDays,
        missingDays,
        totalGapMinutes,
        dateRange: { first: firstDate, last: lastDate },
        weekSummary,
        issueDetails: dayDetails.slice(-20), // Last 20 issues
        overallCoverage: totalDays > 0 ? completeDays / totalDays : 0,
    };
}

function printAssetReport(report) {
    if (!report) return;

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Asset: ${report.category}/${report.asset}`);
    console.log('='.repeat(60));

    console.log(`\nOverall Statistics:`);
    console.log(`  Total Candles: ${formatNumber(report.totalCandles)}`);
    console.log(`  Date Range: ${report.dateRange.first} to ${report.dateRange.last}`);
    console.log(`  Total Days: ${report.totalDays}`);
    console.log(`  Complete Days (>95%): ${report.completeDays} (${formatPercent(report.overallCoverage)})`);
    console.log(`  Partial Days: ${report.partialDays}`);
    console.log(`  Missing/Empty Days: ${report.missingDays}`);
    console.log(`  Total Gap Minutes: ${formatNumber(report.totalGapMinutes)}`);

    console.log(`\nWeek Summary (last 10):`);
    const recentWeeks = report.weekSummary.slice(-10);
    for (const week of recentWeeks) {
        console.log(`  ${week.week}: ${week.days} days, ${formatNumber(week.candles)} candles, ${week.complete}/${week.days} complete`);
    }

    if (report.issueDetails.length > 0) {
        console.log(`\nRecent Days with Issues:`);
        for (const day of report.issueDetails.slice(-5)) {
            console.log(`  ${day.date}: ${day.candles}/${day.expected} (${formatPercent(day.coverage)}), ${day.gaps} gaps (${day.gapMinutes} min)`);
        }
    }
}

function auditAll(filterCategory = null, filterAsset = null) {
    console.log('=' .repeat(60));
    console.log('S.U.P.I.D. Data Quality Audit');
    console.log('='.repeat(60));
    console.log(`Data Directory: ${DATA_DIR}`);
    console.log(`Filter: ${filterCategory || 'All'} / ${filterAsset || 'All'}`);
    console.log('');

    const categories = fs.readdirSync(DATA_DIR)
        .filter(d => {
            const p = path.join(DATA_DIR, d);
            return fs.statSync(p).isDirectory() && !d.startsWith('.');
        });

    const allReports = [];

    for (const category of categories) {
        if (filterCategory && category !== filterCategory) continue;

        const categoryPath = path.join(DATA_DIR, category);
        const assets = fs.readdirSync(categoryPath)
            .filter(d => fs.statSync(path.join(categoryPath, d)).isDirectory());

        for (const asset of assets) {
            if (filterAsset && asset !== filterAsset) continue;

            const report = auditAsset(category, asset);
            if (report) {
                allReports.push(report);
                printAssetReport(report);
            }
        }
    }

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('SUMMARY');
    console.log('='.repeat(60));

    // Sort by total candles
    allReports.sort((a, b) => b.totalCandles - a.totalCandles);

    let grandTotal = 0;
    let grandComplete = 0;
    let grandDays = 0;

    console.log('\nAssets by Data Volume:');
    console.log('-'.repeat(70));
    console.log('Asset'.padEnd(25) + 'Candles'.padStart(15) + 'Days'.padStart(10) + 'Complete'.padStart(12) + 'Coverage'.padStart(10));
    console.log('-'.repeat(70));

    for (const r of allReports) {
        grandTotal += r.totalCandles;
        grandComplete += r.completeDays;
        grandDays += r.totalDays;

        console.log(
            `${r.category}/${r.asset}`.padEnd(25) +
            formatNumber(r.totalCandles).padStart(15) +
            r.totalDays.toString().padStart(10) +
            r.completeDays.toString().padStart(12) +
            formatPercent(r.overallCoverage).padStart(10)
        );
    }

    console.log('-'.repeat(70));
    console.log(
        'TOTAL'.padEnd(25) +
        formatNumber(grandTotal).padStart(15) +
        grandDays.toString().padStart(10) +
        grandComplete.toString().padStart(12) +
        formatPercent(grandDays > 0 ? grandComplete / grandDays : 0).padStart(10)
    );

    // Missing data alerts
    console.log('\n' + '='.repeat(60));
    console.log('DATA GAPS ALERT');
    console.log('='.repeat(60));

    const gapAlerts = allReports.filter(r => r.overallCoverage < 0.8 || r.totalGapMinutes > 1000);
    if (gapAlerts.length > 0) {
        console.log('\nAssets with significant gaps:');
        for (const r of gapAlerts) {
            console.log(`  ${r.category}/${r.asset}: ${formatPercent(r.overallCoverage)} coverage, ${formatNumber(r.totalGapMinutes)} gap minutes`);
        }
    } else {
        console.log('\nNo significant gaps detected!');
    }

    // Recommendations
    console.log('\n' + '='.repeat(60));
    console.log('RECOMMENDATIONS');
    console.log('='.repeat(60));

    const lowData = allReports.filter(r => r.totalCandles < 50000);
    if (lowData.length > 0) {
        console.log('\nAssets needing more historical data (< 50K candles):');
        for (const r of lowData) {
            console.log(`  ${r.category}/${r.asset}: Only ${formatNumber(r.totalCandles)} candles`);
            console.log(`    Run: node polygon_backfill.js "${r.category}" ${r.asset} 730`);
        }
    }

    return allReports;
}

// Main
const args = process.argv.slice(2);
const category = args[0];
const asset = args[1];

auditAll(category, asset);
