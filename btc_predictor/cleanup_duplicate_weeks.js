/**
 * Cleanup Duplicate Week Folders
 *
 * This script:
 * 1. Finds all week_2025-12-02 folders (wrong week due to UTC bug)
 * 2. Merges their data into week_2025-12-01 (correct week)
 * 3. Deletes the empty wrong folders
 * 4. Cleans up any "processed" folders
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');

// Get correct week start (Monday) using UTC
function getWeekStart(date) {
    const d = new Date(date);
    const day = d.getUTCDay();
    const diff = d.getUTCDate() - day + (day === 0 ? -6 : 1);
    d.setUTCDate(diff);
    return d;
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

// Merge two arrays of candles, deduplicating by timestamp
function mergeCandles(existing, newCandles) {
    const map = new Map();

    // Add existing candles
    for (const candle of existing) {
        map.set(candle.timestamp, candle);
    }

    // Add/overwrite with new candles
    for (const candle of newCandles) {
        map.set(candle.timestamp, candle);
    }

    // Convert back to array and sort by timestamp
    return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

function loadJsonFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
    } catch {
        return [];
    }
}

function saveJsonFile(filePath, data) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

// Delete a directory recursively
function deleteDir(dirPath) {
    if (fs.existsSync(dirPath)) {
        fs.readdirSync(dirPath).forEach((file) => {
            const curPath = path.join(dirPath, file);
            if (fs.lstatSync(curPath).isDirectory()) {
                deleteDir(curPath);
            } else {
                fs.unlinkSync(curPath);
            }
        });
        fs.rmdirSync(dirPath);
    }
}

// Find all week_2025-12-02 folders (and any other wrong week folders)
function findWrongWeekFolders() {
    const wrongFolders = [];

    const categories = fs.readdirSync(DATA_DIR).filter(d => {
        const p = path.join(DATA_DIR, d);
        return fs.statSync(p).isDirectory() && !d.startsWith('.');
    });

    for (const category of categories) {
        const categoryPath = path.join(DATA_DIR, category);
        const assets = fs.readdirSync(categoryPath).filter(d => {
            const p = path.join(categoryPath, d);
            return fs.statSync(p).isDirectory();
        });

        for (const asset of assets) {
            const assetPath = path.join(categoryPath, asset);
            const folders = fs.readdirSync(assetPath).filter(d => {
                const p = path.join(assetPath, d);
                return fs.statSync(p).isDirectory() && d.startsWith('week_');
            });

            for (const folder of folders) {
                // Check if this is a wrong week folder (not starting on Monday)
                const weekDateStr = folder.replace('week_', '');
                const weekDate = new Date(weekDateStr + 'T00:00:00Z');
                const dayOfWeek = weekDate.getUTCDay();

                // Monday = 1, if not Monday, it's wrong
                if (dayOfWeek !== 1) {
                    wrongFolders.push({
                        category,
                        asset,
                        wrongFolder: folder,
                        wrongPath: path.join(assetPath, folder),
                        correctWeekStart: formatDate(getWeekStart(weekDate)),
                    });
                }
            }
        }
    }

    return wrongFolders;
}

// Find all "processed" folders
function findProcessedFolders() {
    const processedFolders = [];

    const categories = fs.readdirSync(DATA_DIR).filter(d => {
        const p = path.join(DATA_DIR, d);
        return fs.statSync(p).isDirectory() && !d.startsWith('.');
    });

    for (const category of categories) {
        const categoryPath = path.join(DATA_DIR, category);
        const assets = fs.readdirSync(categoryPath).filter(d => {
            const p = path.join(categoryPath, d);
            return fs.statSync(p).isDirectory();
        });

        for (const asset of assets) {
            const assetPath = path.join(categoryPath, asset);
            const processedPath = path.join(assetPath, 'processed');

            if (fs.existsSync(processedPath)) {
                processedFolders.push({
                    category,
                    asset,
                    path: processedPath,
                });
            }
        }
    }

    return processedFolders;
}

// Merge data from wrong folder to correct folder
function mergeWrongFolder(wrongInfo) {
    const { category, asset, wrongFolder, wrongPath, correctWeekStart } = wrongInfo;
    const assetPath = path.dirname(wrongPath);
    const correctPath = path.join(assetPath, `week_${correctWeekStart}`);

    // Get all JSON files in wrong folder
    const files = fs.readdirSync(wrongPath).filter(f => f.endsWith('.json'));

    let totalMerged = 0;

    for (const file of files) {
        const wrongFilePath = path.join(wrongPath, file);
        const correctFilePath = path.join(correctPath, file);

        const wrongData = loadJsonFile(wrongFilePath);
        const existingData = loadJsonFile(correctFilePath);

        if (wrongData.length > 0) {
            const merged = mergeCandles(existingData, wrongData);
            saveJsonFile(correctFilePath, merged);
            totalMerged += wrongData.length;
        }
    }

    // Delete the wrong folder after merging
    deleteDir(wrongPath);

    return totalMerged;
}

// Merge processed folder data to correct locations
function mergeProcessedFolder(info) {
    const { category, asset, path: processedPath } = info;
    const assetPath = path.dirname(processedPath);

    let totalMerged = 0;

    // Get all week folders in processed
    const weekFolders = fs.readdirSync(processedPath).filter(d => {
        const p = path.join(processedPath, d);
        return fs.statSync(p).isDirectory() && d.startsWith('week_');
    });

    for (const weekFolder of weekFolders) {
        const srcWeekPath = path.join(processedPath, weekFolder);
        const dstWeekPath = path.join(assetPath, weekFolder);

        const files = fs.readdirSync(srcWeekPath).filter(f => f.endsWith('.json'));

        for (const file of files) {
            const srcFilePath = path.join(srcWeekPath, file);
            const dstFilePath = path.join(dstWeekPath, file);

            const srcData = loadJsonFile(srcFilePath);
            const existingData = loadJsonFile(dstFilePath);

            if (srcData.length > 0) {
                const merged = mergeCandles(existingData, srcData);
                saveJsonFile(dstFilePath, merged);
                totalMerged += srcData.length;
            }
        }
    }

    // Delete processed folder after merging
    deleteDir(processedPath);

    return totalMerged;
}

// Main
async function main() {
    console.log('='.repeat(60));
    console.log('Cleanup Duplicate Week Folders');
    console.log('='.repeat(60));
    console.log(`Data directory: ${DATA_DIR}`);
    console.log('');

    // Find wrong week folders
    console.log('Finding wrong week folders...');
    const wrongFolders = findWrongWeekFolders();
    console.log(`Found ${wrongFolders.length} wrong week folders\n`);

    // Merge wrong folders
    let totalMergedFromWrong = 0;
    for (const info of wrongFolders) {
        console.log(`Merging ${info.category}/${info.asset}/${info.wrongFolder} -> week_${info.correctWeekStart}`);
        const merged = mergeWrongFolder(info);
        totalMergedFromWrong += merged;
    }
    console.log(`\nMerged ${totalMergedFromWrong} candles from wrong week folders\n`);

    // Find processed folders
    console.log('Finding processed folders...');
    const processedFolders = findProcessedFolders();
    console.log(`Found ${processedFolders.length} processed folders\n`);

    // Merge processed folders
    let totalMergedFromProcessed = 0;
    for (const info of processedFolders) {
        console.log(`Merging ${info.category}/${info.asset}/processed`);
        const merged = mergeProcessedFolder(info);
        totalMergedFromProcessed += merged;
    }
    console.log(`\nMerged ${totalMergedFromProcessed} candles from processed folders\n`);

    // Summary
    console.log('='.repeat(60));
    console.log('CLEANUP COMPLETE');
    console.log('='.repeat(60));
    console.log(`Wrong week folders fixed: ${wrongFolders.length}`);
    console.log(`Processed folders cleaned: ${processedFolders.length}`);
    console.log(`Total candles merged: ${totalMergedFromWrong + totalMergedFromProcessed}`);
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
