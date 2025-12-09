/**
 * Scan Data Problems
 *
 * Scans all week folders in Data/ directory and detects:
 * 1. Weeks with more than 7 days (should be max 7)
 * 2. Empty week folders (0 files)
 * 3. Misplaced day files (files in wrong week folder)
 * 4. Non-Monday week folder names
 *
 * Usage:
 *   node scan_data_problems.js              # Scan all data
 *   node scan_data_problems.js --fix        # Scan and auto-fix
 *   node scan_data_problems.js Crypto BTC   # Scan specific asset
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');
const REPORT_FILE = path.join(__dirname, 'data_problems_report.json');

// Command line args
const args = process.argv.slice(2);
const FIX_MODE = args.includes('--fix');
const nonFlagArgs = args.filter(a => !a.startsWith('--'));
const TARGET_CATEGORY = nonFlagArgs[0] || null;
const TARGET_ASSET = nonFlagArgs[1] || null;

// Day names for display
const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

// Results storage
const results = {
    scannedAt: new Date().toISOString(),
    totalCategories: 0,
    totalAssets: 0,
    totalWeeks: 0,
    totalDayFiles: 0,
    problems: {
        weeksWithTooManyDays: [],
        emptyWeeks: [],
        misplacedFiles: [],
        nonMondayWeeks: []
    },
    summary: {
        weeksWithTooManyDays: 0,
        emptyWeeks: 0,
        misplacedFiles: 0,
        nonMondayWeeks: 0
    }
};

/**
 * Get the correct week start (Monday) for any date - FIXED VERSION
 */
function getCorrectWeekStart(date) {
    const d = new Date(date);
    d.setUTCHours(0, 0, 0, 0);
    const day = d.getUTCDay();
    const diff = day === 0 ? -6 : 1 - day;  // Fixed: use relative diff
    d.setUTCDate(d.getUTCDate() + diff);
    return d;
}

/**
 * Format date as YYYY-MM-DD
 */
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

/**
 * Check if a date string represents a Monday
 */
function isMonday(dateStr) {
    const d = new Date(dateStr + 'T00:00:00Z');
    return d.getUTCDay() === 1;
}

/**
 * Get day of week name for a date string
 */
function getDayName(dateStr) {
    const d = new Date(dateStr + 'T00:00:00Z');
    return DAY_NAMES[d.getUTCDay()];
}

/**
 * Check if a day file belongs to a specific week folder
 */
function dayBelongsToWeek(dayStr, weekStr) {
    const dayDate = new Date(dayStr + 'T00:00:00Z');
    const expectedWeekStart = getCorrectWeekStart(dayDate);
    const expectedWeekStr = formatDate(expectedWeekStart);
    return weekStr === expectedWeekStr;
}

/**
 * Get all categories in Data directory
 */
function getCategories() {
    if (!fs.existsSync(DATA_DIR)) return [];
    return fs.readdirSync(DATA_DIR).filter(f => {
        const stat = fs.statSync(path.join(DATA_DIR, f));
        return stat.isDirectory();
    });
}

/**
 * Get all assets in a category
 */
function getAssets(category) {
    const catPath = path.join(DATA_DIR, category);
    if (!fs.existsSync(catPath)) return [];
    return fs.readdirSync(catPath).filter(f => {
        const stat = fs.statSync(path.join(catPath, f));
        return stat.isDirectory();
    });
}

/**
 * Get all week folders for an asset
 */
function getWeekFolders(category, asset) {
    const assetPath = path.join(DATA_DIR, category, asset);
    if (!fs.existsSync(assetPath)) return [];
    return fs.readdirSync(assetPath).filter(f => {
        return f.startsWith('week_') && fs.statSync(path.join(assetPath, f)).isDirectory();
    });
}

/**
 * Get all day files in a week folder
 */
function getDayFiles(category, asset, week) {
    const weekPath = path.join(DATA_DIR, category, asset, week);
    if (!fs.existsSync(weekPath)) return [];
    return fs.readdirSync(weekPath).filter(f => f.endsWith('.json'));
}

/**
 * Move a file from one week folder to another
 */
function moveFile(category, asset, fromWeek, toWeek, dayFile) {
    const fromPath = path.join(DATA_DIR, category, asset, fromWeek, dayFile);
    const toDir = path.join(DATA_DIR, category, asset, toWeek);
    const toPath = path.join(toDir, dayFile);

    // Create target directory if it doesn't exist
    if (!fs.existsSync(toDir)) {
        fs.mkdirSync(toDir, { recursive: true });
    }

    // If file already exists in target, merge the data
    if (fs.existsSync(toPath)) {
        try {
            const existingData = JSON.parse(fs.readFileSync(toPath, 'utf8'));
            const newData = JSON.parse(fs.readFileSync(fromPath, 'utf8'));

            // Merge by timestamp
            const map = new Map();
            for (const candle of existingData) {
                map.set(candle.timestamp, candle);
            }
            for (const candle of newData) {
                map.set(candle.timestamp, candle);
            }
            const merged = Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);

            fs.writeFileSync(toPath, JSON.stringify(merged, null, 2));
            fs.unlinkSync(fromPath);
            return { action: 'merged', count: merged.length };
        } catch (e) {
            return { action: 'error', error: e.message };
        }
    } else {
        // Simply move the file
        fs.renameSync(fromPath, toPath);
        return { action: 'moved' };
    }
}

/**
 * Remove empty directory
 */
function removeEmptyDir(dirPath) {
    if (fs.existsSync(dirPath)) {
        const files = fs.readdirSync(dirPath);
        if (files.length === 0) {
            fs.rmdirSync(dirPath);
            return true;
        }
    }
    return false;
}

/**
 * Scan a single asset for problems
 */
function scanAsset(category, asset) {
    const weeks = getWeekFolders(category, asset);
    let assetProblems = {
        tooManyDays: [],
        emptyWeeks: [],
        misplaced: [],
        nonMonday: []
    };

    for (const week of weeks) {
        results.totalWeeks++;

        // Extract week date from folder name (week_YYYY-MM-DD)
        const weekDateStr = week.replace('week_', '');
        const weekPath = `${category}/${asset}/${week}`;

        // Check 1: Is the week folder named after a Monday?
        if (!isMonday(weekDateStr)) {
            assetProblems.nonMonday.push({
                path: weekPath,
                weekDate: weekDateStr,
                dayOfWeek: getDayName(weekDateStr)
            });
        }

        // Get day files in this week
        const dayFiles = getDayFiles(category, asset, week);
        results.totalDayFiles += dayFiles.length;

        // Check 2: Empty week folders
        if (dayFiles.length === 0) {
            assetProblems.emptyWeeks.push({
                path: weekPath
            });
            continue;
        }

        // Check 3: Too many days (>7)
        if (dayFiles.length > 7) {
            assetProblems.tooManyDays.push({
                path: weekPath,
                dayCount: dayFiles.length,
                days: dayFiles.map(f => f.replace('.json', '')).sort()
            });
        }

        // Check 4: Misplaced files
        for (const dayFile of dayFiles) {
            const dayStr = dayFile.replace('.json', '');
            if (!dayBelongsToWeek(dayStr, weekDateStr)) {
                const correctWeekStart = getCorrectWeekStart(new Date(dayStr + 'T00:00:00Z'));
                const correctWeekStr = formatDate(correctWeekStart);

                assetProblems.misplaced.push({
                    file: dayFile,
                    currentPath: weekPath,
                    inWeek: weekDateStr,
                    shouldBeIn: correctWeekStr,
                    correctPath: `${category}/${asset}/week_${correctWeekStr}`
                });
            }
        }
    }

    return assetProblems;
}

/**
 * Fix problems for a single asset
 */
function fixAssetProblems(category, asset, problems) {
    const fixes = {
        movedFiles: 0,
        mergedFiles: 0,
        removedEmptyDirs: 0,
        errors: []
    };

    // Fix misplaced files first
    for (const misplaced of problems.misplaced) {
        const fromWeek = `week_${misplaced.inWeek}`;
        const toWeek = `week_${misplaced.shouldBeIn}`;

        const result = moveFile(category, asset, fromWeek, toWeek, misplaced.file);

        if (result.action === 'moved') {
            fixes.movedFiles++;
            console.log(`  Moved ${misplaced.file}: ${fromWeek} -> ${toWeek}`);
        } else if (result.action === 'merged') {
            fixes.mergedFiles++;
            console.log(`  Merged ${misplaced.file}: ${fromWeek} -> ${toWeek} (${result.count} candles)`);
        } else {
            fixes.errors.push({ file: misplaced.file, error: result.error });
            console.log(`  ERROR ${misplaced.file}: ${result.error}`);
        }
    }

    // Remove empty week directories
    for (const emptyWeek of problems.emptyWeeks) {
        const weekPath = path.join(DATA_DIR, emptyWeek.path);
        if (removeEmptyDir(weekPath)) {
            fixes.removedEmptyDirs++;
            console.log(`  Removed empty: ${emptyWeek.path}`);
        }
    }

    // Check non-Monday folders for remaining files and clean up if empty
    for (const nonMonday of problems.nonMonday) {
        const weekPath = path.join(DATA_DIR, nonMonday.path);
        if (removeEmptyDir(weekPath)) {
            fixes.removedEmptyDirs++;
            console.log(`  Removed non-Monday empty: ${nonMonday.path}`);
        }
    }

    return fixes;
}

/**
 * Main scanning function
 */
async function main() {
    console.log('='.repeat(60));
    console.log('Data Problem Scanner');
    console.log('='.repeat(60));
    console.log(`Mode: ${FIX_MODE ? 'SCAN + FIX' : 'SCAN ONLY'}`);
    console.log(`Target: ${TARGET_CATEGORY ? `${TARGET_CATEGORY}/${TARGET_ASSET || '*'}` : 'All assets'}`);
    console.log('');

    const startTime = Date.now();
    const categories = TARGET_CATEGORY ? [TARGET_CATEGORY] : getCategories();
    results.totalCategories = categories.length;

    let totalFixes = { movedFiles: 0, mergedFiles: 0, removedEmptyDirs: 0, errors: [] };

    for (const category of categories) {
        console.log(`\nScanning ${category}...`);

        const assets = TARGET_ASSET ? [TARGET_ASSET] : getAssets(category);

        for (const asset of assets) {
            results.totalAssets++;
            process.stdout.write(`  ${asset}...`);

            const problems = scanAsset(category, asset);

            // Count problems
            const problemCount = problems.tooManyDays.length + problems.emptyWeeks.length +
                                problems.misplaced.length + problems.nonMonday.length;

            if (problemCount > 0) {
                console.log(` ${problemCount} problems found`);

                // Add to results
                results.problems.weeksWithTooManyDays.push(...problems.tooManyDays);
                results.problems.emptyWeeks.push(...problems.emptyWeeks);
                results.problems.misplacedFiles.push(...problems.misplaced);
                results.problems.nonMondayWeeks.push(...problems.nonMonday);

                // Fix if in fix mode
                if (FIX_MODE && problemCount > 0) {
                    const fixes = fixAssetProblems(category, asset, problems);
                    totalFixes.movedFiles += fixes.movedFiles;
                    totalFixes.mergedFiles += fixes.mergedFiles;
                    totalFixes.removedEmptyDirs += fixes.removedEmptyDirs;
                    totalFixes.errors.push(...fixes.errors);
                }
            } else {
                console.log(' OK');
            }
        }
    }

    // Update summary counts
    results.summary.weeksWithTooManyDays = results.problems.weeksWithTooManyDays.length;
    results.summary.emptyWeeks = results.problems.emptyWeeks.length;
    results.summary.misplacedFiles = results.problems.misplacedFiles.length;
    results.summary.nonMondayWeeks = results.problems.nonMondayWeeks.length;

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    // Print summary
    console.log('\n' + '='.repeat(60));
    console.log('SCAN SUMMARY');
    console.log('='.repeat(60));
    console.log(`Categories scanned: ${results.totalCategories}`);
    console.log(`Assets scanned: ${results.totalAssets}`);
    console.log(`Week folders scanned: ${results.totalWeeks}`);
    console.log(`Day files scanned: ${results.totalDayFiles.toLocaleString()}`);
    console.log(`Time elapsed: ${elapsed}s`);
    console.log('');
    console.log('PROBLEMS FOUND:');
    console.log(`  Weeks with >7 days: ${results.summary.weeksWithTooManyDays}`);
    console.log(`  Empty weeks: ${results.summary.emptyWeeks}`);
    console.log(`  Misplaced files: ${results.summary.misplacedFiles}`);
    console.log(`  Non-Monday weeks: ${results.summary.nonMondayWeeks}`);

    const totalProblems = results.summary.weeksWithTooManyDays + results.summary.emptyWeeks +
                         results.summary.misplacedFiles + results.summary.nonMondayWeeks;

    if (totalProblems === 0) {
        console.log('\nAll data is clean! No problems found.');
    } else {
        console.log(`\nTotal problems: ${totalProblems}`);

        if (FIX_MODE) {
            console.log('\nFIXES APPLIED:');
            console.log(`  Files moved: ${totalFixes.movedFiles}`);
            console.log(`  Files merged: ${totalFixes.mergedFiles}`);
            console.log(`  Empty dirs removed: ${totalFixes.removedEmptyDirs}`);
            if (totalFixes.errors.length > 0) {
                console.log(`  Errors: ${totalFixes.errors.length}`);
            }
        } else {
            console.log('\nRun with --fix to automatically fix these issues:');
            console.log('  node scan_data_problems.js --fix');
        }
    }

    // Save report
    fs.writeFileSync(REPORT_FILE, JSON.stringify(results, null, 2));
    console.log(`\nFull report saved to: ${REPORT_FILE}`);
    console.log('='.repeat(60));

    // Show some examples of problems
    if (results.problems.weeksWithTooManyDays.length > 0) {
        console.log('\nExample weeks with >7 days:');
        for (const problem of results.problems.weeksWithTooManyDays.slice(0, 3)) {
            console.log(`  ${problem.path}: ${problem.dayCount} days`);
            console.log(`    Days: ${problem.days.join(', ')}`);
        }
    }

    if (results.problems.misplacedFiles.length > 0) {
        console.log('\nExample misplaced files:');
        for (const problem of results.problems.misplacedFiles.slice(0, 5)) {
            console.log(`  ${problem.file} in ${problem.inWeek} should be in ${problem.shouldBeIn}`);
        }
    }
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
