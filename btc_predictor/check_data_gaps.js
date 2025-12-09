/**
 * Data Gap Detection Utility
 *
 * Scans the Data folder and reports:
 * - Missing days within weeks
 * - Weeks with incomplete data
 * - Assets with insufficient data for training
 *
 * Usage: node check_data_gaps.js [category] [asset]
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');

// Get all dates between two dates
function getDatesBetween(startDate, endDate) {
    const dates = [];
    const current = new Date(startDate);
    const end = new Date(endDate);

    while (current <= end) {
        dates.push(current.toISOString().split('T')[0]);
        current.setDate(current.getDate() + 1);
    }

    return dates;
}

// Get week start (Monday)
function getWeekStart(dateStr) {
    const date = new Date(dateStr);
    const day = date.getUTCDay();
    const diff = date.getUTCDate() - day + (day === 0 ? -6 : 1);
    date.setUTCDate(diff);
    return date.toISOString().split('T')[0];
}

// Analyze a single asset
function analyzeAsset(category, asset) {
    const assetPath = path.join(DATA_DIR, category, asset);

    if (!fs.existsSync(assetPath)) {
        return { error: 'Asset directory not found' };
    }

    const weekDirs = fs.readdirSync(assetPath)
        .filter(d => d.startsWith('week_'))
        .sort();

    if (weekDirs.length === 0) {
        return { error: 'No data found' };
    }

    const results = {
        totalWeeks: weekDirs.length,
        totalFiles: 0,
        totalCandles: 0,
        firstDate: null,
        lastDate: null,
        missingDays: [],
        incompleteWeeks: [],
        weekDetails: [],
    };

    let allDates = [];

    for (const weekDir of weekDirs) {
        const weekPath = path.join(assetPath, weekDir);
        const weekStart = weekDir.replace('week_', '');

        const files = fs.readdirSync(weekPath)
            .filter(f => f.endsWith('.json'))
            .sort();

        let weekCandles = 0;
        const weekDates = [];

        for (const file of files) {
            const dateStr = file.replace('.json', '');
            weekDates.push(dateStr);
            allDates.push(dateStr);

            try {
                const data = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));
                weekCandles += data.length;
                results.totalCandles += data.length;
            } catch {
                // Skip invalid files
            }
        }

        results.totalFiles += files.length;

        // Check for missing days in this week
        if (weekDates.length > 0 && weekDates.length < 7) {
            // For markets that trade 24/7 (crypto), expect 7 days
            // For stocks/commodities, expect 5 days (Mon-Fri)
            const isCrypto = category === 'Crypto';
            const expectedDays = isCrypto ? 7 : 5;

            if (weekDates.length < expectedDays) {
                results.incompleteWeeks.push({
                    week: weekStart,
                    daysFound: weekDates.length,
                    expected: expectedDays,
                    dates: weekDates,
                });
            }
        }

        results.weekDetails.push({
            week: weekStart,
            files: files.length,
            candles: weekCandles,
            dates: weekDates,
        });
    }

    // Sort all dates and find gaps
    allDates.sort();
    if (allDates.length > 0) {
        results.firstDate = allDates[0];
        results.lastDate = allDates[allDates.length - 1];

        // Find missing days
        const allExpectedDates = getDatesBetween(results.firstDate, results.lastDate);
        const existingDates = new Set(allDates);

        for (const date of allExpectedDates) {
            if (!existingDates.has(date)) {
                // For stocks, skip weekends
                const dayOfWeek = new Date(date).getDay();
                const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
                const isCrypto = category === 'Crypto';

                if (isCrypto || !isWeekend) {
                    results.missingDays.push(date);
                }
            }
        }
    }

    return results;
}

// Generate full report
function generateReport(specificCategory = null, specificAsset = null) {
    console.log('============================================================');
    console.log('Data Gap Detection Report');
    console.log('============================================================');
    console.log(`Data directory: ${DATA_DIR}`);
    console.log(`Generated: ${new Date().toISOString()}`);
    console.log('============================================================\n');

    const categories = specificCategory
        ? [specificCategory]
        : ['Crypto', 'Stock Market', 'Commodities', 'Currencies'];

    const summary = {
        totalAssets: 0,
        healthyAssets: 0,
        assetsWithGaps: 0,
        criticalAssets: [],
        totalCandles: 0,
    };

    for (const category of categories) {
        const categoryPath = path.join(DATA_DIR, category);
        if (!fs.existsSync(categoryPath)) {
            console.log(`Category not found: ${category}\n`);
            continue;
        }

        console.log(`\n=== ${category} ===\n`);

        const assets = specificAsset
            ? [specificAsset]
            : fs.readdirSync(categoryPath).filter(f =>
                fs.statSync(path.join(categoryPath, f)).isDirectory()
            );

        for (const asset of assets) {
            const analysis = analyzeAsset(category, asset);
            summary.totalAssets++;

            if (analysis.error) {
                console.log(`${asset}: ${analysis.error}`);
                summary.criticalAssets.push({ category, asset, reason: analysis.error });
                continue;
            }

            summary.totalCandles += analysis.totalCandles;

            const status = [];
            if (analysis.totalCandles < 10000) {
                status.push(`LOW DATA (${analysis.totalCandles} candles)`);
            }
            if (analysis.missingDays.length > 0) {
                status.push(`${analysis.missingDays.length} missing days`);
            }
            if (analysis.incompleteWeeks.length > 0) {
                status.push(`${analysis.incompleteWeeks.length} incomplete weeks`);
            }

            if (status.length === 0) {
                summary.healthyAssets++;
                console.log(`${asset}: OK (${analysis.totalCandles.toLocaleString()} candles, ${analysis.totalWeeks} weeks)`);
            } else {
                summary.assetsWithGaps++;
                console.log(`${asset}: ${status.join(', ')}`);

                if (analysis.totalCandles < 1000) {
                    summary.criticalAssets.push({
                        category,
                        asset,
                        reason: `Only ${analysis.totalCandles} candles`,
                    });
                }

                // Show details for problematic assets
                if (analysis.missingDays.length > 0 && analysis.missingDays.length <= 10) {
                    console.log(`  Missing days: ${analysis.missingDays.join(', ')}`);
                } else if (analysis.missingDays.length > 10) {
                    console.log(`  Missing days: ${analysis.missingDays.slice(0, 5).join(', ')}... and ${analysis.missingDays.length - 5} more`);
                }

                for (const week of analysis.incompleteWeeks.slice(0, 3)) {
                    console.log(`  Week ${week.week}: ${week.daysFound}/${week.expected} days`);
                }
            }
        }
    }

    // Summary
    console.log('\n============================================================');
    console.log('SUMMARY');
    console.log('============================================================');
    console.log(`Total assets scanned: ${summary.totalAssets}`);
    console.log(`Healthy assets: ${summary.healthyAssets}`);
    console.log(`Assets with gaps: ${summary.assetsWithGaps}`);
    console.log(`Total candles: ${summary.totalCandles.toLocaleString()}`);

    if (summary.criticalAssets.length > 0) {
        console.log('\nCRITICAL - Need attention:');
        for (const critical of summary.criticalAssets) {
            console.log(`  ${critical.category}/${critical.asset}: ${critical.reason}`);
        }
    }

    console.log('\n============================================================');

    return summary;
}

// Main
const args = process.argv.slice(2);
const category = args[0] || null;
const asset = args[1] || null;

generateReport(category, asset);
