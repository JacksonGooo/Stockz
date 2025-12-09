/**
 * Live Data Collector - TradingView
 * Collects real-time BTC data every minute
 * Saves to daily files in weekly folders
 * Run this continuously to build up data over time
 */

const TradingView = require('@mathieuc/tradingview');
const fs = require('fs');
const path = require('path');

// Configuration
const SYMBOL = 'COINBASE:BTCUSD';
const DATA_DIR = path.join(__dirname, 'data', 'live');

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

function formatTime(date) {
    return date.toISOString().replace('T', ' ').split('.')[0];
}

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

function saveCandle(candle) {
    const date = new Date(candle.timestamp);
    const dayKey = formatDate(date);

    // Get week folder
    const weekStart = getWeekStart(date);
    const weekFolder = path.join(DATA_DIR, `week_${formatDate(weekStart)}`);
    ensureDir(weekFolder);

    // Load or create day file
    const dayFile = path.join(weekFolder, `${dayKey}.json`);
    let dayData = [];

    if (fs.existsSync(dayFile)) {
        dayData = JSON.parse(fs.readFileSync(dayFile, 'utf8'));
    }

    // Check if candle already exists (by timestamp)
    const exists = dayData.some(c => c.timestamp === candle.timestamp);
    if (!exists) {
        dayData.push(candle);
        dayData.sort((a, b) => a.timestamp - b.timestamp);
        fs.writeFileSync(dayFile, JSON.stringify(dayData, null, 2));
        return true;
    }
    return false;
}

async function startLiveCollection() {
    console.log('============================================================');
    console.log('Live Data Collector - TradingView');
    console.log('============================================================');
    console.log(`Symbol: ${SYMBOL}`);
    console.log(`Data directory: ${DATA_DIR}`);
    console.log('============================================================');
    console.log('Collecting real-time data. Press Ctrl+C to stop.');
    console.log('============================================================\n');

    ensureDir(DATA_DIR);

    const client = new TradingView.Client();
    const chart = new client.Session.Chart();

    // Set market
    chart.setMarket(SYMBOL, {
        timeframe: '1',  // 1 minute
    });

    let lastCandle = null;
    let candlesCollected = 0;
    let startTime = Date.now();

    // Handle updates
    chart.onUpdate(() => {
        const periods = chart.periods;
        if (!periods || periods.length === 0) return;

        // Get the latest candle
        const latest = periods[periods.length - 1];

        // Check if it's a new candle
        if (lastCandle === null || latest.time !== lastCandle.time) {
            const candle = {
                timestamp: latest.time * 1000,
                open: latest.open,
                high: latest.max,
                low: latest.min,
                close: latest.close,
                volume: latest.volume || 0
            };

            const saved = saveCandle(candle);
            if (saved) {
                candlesCollected++;
                const date = new Date(candle.timestamp);
                const runtime = Math.floor((Date.now() - startTime) / 1000 / 60);
                console.log(`[${formatTime(date)}] BTC: $${candle.close.toFixed(2)} | Candles: ${candlesCollected} | Runtime: ${runtime}m`);
            }

            lastCandle = latest;
        }
    });

    // Handle errors
    chart.onError((err) => {
        console.log('Error:', err);
    });

    // Keep alive
    console.log('Waiting for data...\n');

    // Graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n\nShutting down...');
        console.log(`Total candles collected: ${candlesCollected}`);
        client.end();
        process.exit(0);
    });

    // Keep the process running
    await new Promise(() => {});
}

// Show status periodically
setInterval(() => {
    // Count files
    if (fs.existsSync(DATA_DIR)) {
        const weeks = fs.readdirSync(DATA_DIR).filter(f => f.startsWith('week_'));
        let totalCandles = 0;

        for (const week of weeks) {
            const weekPath = path.join(DATA_DIR, week);
            const days = fs.readdirSync(weekPath).filter(f => f.endsWith('.json'));
            for (const day of days) {
                const dayData = JSON.parse(fs.readFileSync(path.join(weekPath, day), 'utf8'));
                totalCandles += dayData.length;
            }
        }

        console.log(`\n--- Status: ${weeks.length} weeks, ${totalCandles.toLocaleString()} total candles ---\n`);
    }
}, 60000);  // Every minute

startLiveCollection().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
