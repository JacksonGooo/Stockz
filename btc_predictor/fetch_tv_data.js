/**
 * Bitcoin Data Fetcher - TradingView API (Mathieu2301)
 * Fetches historical BTC 1-minute data from TradingView
 * Requests extended historical data for more candles
 */

const TradingView = require('@mathieuc/tradingview');
const fs = require('fs');
const path = require('path');

// Configuration
const SYMBOL = 'COINBASE:BTCUSD';
const TIMEFRAME = '1';  // 1 minute
const TARGET_CANDLES = 1000000;  // Request a LOT of candles

// Helper functions
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

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

async function fetchHistoricalData() {
    console.log('============================================================');
    console.log('Bitcoin Data Fetcher - TradingView API');
    console.log('============================================================');
    console.log(`Symbol: ${SYMBOL}`);
    console.log(`Timeframe: ${TIMEFRAME} minute`);
    console.log(`Target: As many candles as possible`);
    console.log('============================================================\n');

    const dataDir = path.join(__dirname, 'data');
    ensureDir(dataDir);

    console.log('[1/3] Connecting to TradingView...');

    const client = new TradingView.Client();
    const chart = new client.Session.Chart();

    // Set market with extended hours for more data
    chart.setMarket(SYMBOL, {
        timeframe: TIMEFRAME,
        range: TARGET_CANDLES,  // Request lots of bars
    });

    console.log('[2/3] Waiting for data (this may take a moment)...');

    // Wait for data to load
    let allPeriods = [];
    let lastCount = 0;
    let stableCount = 0;

    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            console.log(`\nTimeout reached. Got ${allPeriods.length} candles.`);
            resolve();
        }, 60000);  // 60 second timeout

        chart.onUpdate(() => {
            allPeriods = chart.periods || [];

            if (allPeriods.length > 0) {
                if (allPeriods.length === lastCount) {
                    stableCount++;
                    if (stableCount > 5) {
                        clearTimeout(timeout);
                        resolve();
                    }
                } else {
                    stableCount = 0;
                    lastCount = allPeriods.length;
                    process.stdout.write(`\r  Received: ${allPeriods.length.toLocaleString()} candles...`);
                }
            }
        });

        chart.onError((err) => {
            clearTimeout(timeout);
            console.log('\nError:', err);
            resolve();
        });
    });

    console.log(`\n\n[3/3] Processing ${allPeriods.length.toLocaleString()} candles...`);

    if (!allPeriods || allPeriods.length === 0) {
        console.log('No data received!');
        client.end();
        return;
    }

    // Process candles by day
    const candlesByDay = {};

    for (const candle of allPeriods) {
        const date = new Date(candle.time * 1000);
        const dayKey = formatDate(date);

        if (!candlesByDay[dayKey]) {
            candlesByDay[dayKey] = [];
        }

        candlesByDay[dayKey].push({
            timestamp: candle.time * 1000,
            open: candle.open,
            high: candle.max,
            low: candle.min,
            close: candle.close,
            volume: candle.volume || 0
        });
    }

    // Save each day
    const days = Object.keys(candlesByDay).sort();
    console.log(`\nOrganizing ${days.length} days into weekly folders...\n`);

    let allCandles = [];
    let filesCreated = 0;
    const weeksCreated = new Set();

    for (const day of days) {
        const dayData = candlesByDay[day];
        const dayDate = new Date(day);

        const weekStart = getWeekStart(dayDate);
        const weekFolder = path.join(dataDir, `week_${formatDate(weekStart)}`);
        ensureDir(weekFolder);
        weeksCreated.add(weekFolder);

        const dayFile = path.join(weekFolder, `${day}.json`);
        fs.writeFileSync(dayFile, JSON.stringify(dayData, null, 2));

        filesCreated++;
        console.log(`  [OK] ${day}: ${dayData.length} candles`);

        allCandles = allCandles.concat(dayData);
    }

    // Save master file
    const masterFile = path.join(dataDir, 'btc_raw_data.json');
    fs.writeFileSync(masterFile, JSON.stringify(allCandles, null, 2));

    const masterSizeMB = (fs.statSync(masterFile).size / (1024 * 1024)).toFixed(2);

    // Summary
    console.log('\n============================================================');
    console.log('DATA FETCH COMPLETE!');
    console.log('============================================================');
    console.log(`Total candles: ${allCandles.length.toLocaleString()}`);
    console.log(`Daily files: ${filesCreated}`);
    console.log(`Weekly folders: ${weeksCreated.size}`);
    console.log(`Master file: ${masterSizeMB} MB`);
    console.log(`Date range: ${days[0]} to ${days[days.length - 1]}`);
    console.log('\n============================================================');
    console.log('Now run: py fetch_data.py');
    console.log('============================================================');

    client.end();
}

fetchHistoricalData().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
