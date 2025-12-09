/**
 * Backfill Data Gaps - TradingView
 *
 * Fetches historical data to fill gaps when the collector was not running.
 * TradingView provides ~300 historical candles when connecting to a chart.
 *
 * Usage:
 *   node backfill_gaps.js           - Backfill all assets
 *   node backfill_gaps.js Crypto    - Backfill specific category
 *   node backfill_gaps.js Crypto BTC - Backfill specific asset
 */

const TradingView = require('@mathieuc/tradingview');
const fs = require('fs');
const path = require('path');

const ASSETS = {
    Crypto: {
        BTC: 'COINBASE:BTCUSD',
        ETH: 'COINBASE:ETHUSD',
        SOL: 'COINBASE:SOLUSD',
        XRP: 'COINBASE:XRPUSD',
        DOGE: 'COINBASE:DOGEUSD',
        ADA: 'COINBASE:ADAUSD',
        AVAX: 'COINBASE:AVAXUSD',
        DOT: 'COINBASE:DOTUSD',
        POL: 'COINBASE:POLUSD',
        LINK: 'COINBASE:LINKUSD',
        LTC: 'COINBASE:LTCUSD',
        UNI: 'COINBASE:UNIUSD',
        ATOM: 'COINBASE:ATOMUSD',
        XLM: 'COINBASE:XLMUSD',
        ALGO: 'COINBASE:ALGOUSD',
    },
    "Stock Market": {
        SPY: 'AMEX:SPY',
        QQQ: 'NASDAQ:QQQ',
        DIA: 'AMEX:DIA',
        IWM: 'AMEX:IWM',
        AAPL: 'NASDAQ:AAPL',
        MSFT: 'NASDAQ:MSFT',
        GOOGL: 'NASDAQ:GOOGL',
        AMZN: 'NASDAQ:AMZN',
        NVDA: 'NASDAQ:NVDA',
        META: 'NASDAQ:META',
        TSLA: 'NASDAQ:TSLA',
        JPM: 'NYSE:JPM',
        V: 'NYSE:V',
        MA: 'NYSE:MA',
        BAC: 'NYSE:BAC',
        UNH: 'NYSE:UNH',
        JNJ: 'NYSE:JNJ',
        WMT: 'NYSE:WMT',
        PG: 'NYSE:PG',
        XOM: 'NYSE:XOM',
    },
    Commodities: {
        GOLD: 'TVC:GOLD',
        SILVER: 'TVC:SILVER',
        OIL: 'TVC:USOIL',
        NATGAS: 'NYMEX:NG1!',
        COPPER: 'COMEX:HG1!',
        PLATINUM: 'TVC:PLATINUM',
    },
    Currencies: {
        EURUSD: 'FX:EURUSD',
        GBPUSD: 'FX:GBPUSD',
        USDJPY: 'FX:USDJPY',
        AUDUSD: 'FX:AUDUSD',
        USDCAD: 'FX:USDCAD',
        USDCHF: 'FX:USDCHF',
        NZDUSD: 'FX:NZDUSD',
        EURGBP: 'FX:EURGBP',
    },
};

const DATA_DIR = path.join(__dirname, '..', 'Data');

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

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

// Load existing candles for a day
function loadDayCandles(category, asset, dateStr) {
    const assetDir = path.join(DATA_DIR, category, asset);
    const weekDirs = fs.readdirSync(assetDir).filter(d => d.startsWith('week_'));

    for (const weekDir of weekDirs) {
        const dayFile = path.join(assetDir, weekDir, `${dateStr}.json`);
        if (fs.existsSync(dayFile)) {
            try {
                return JSON.parse(fs.readFileSync(dayFile, 'utf8'));
            } catch {
                return [];
            }
        }
    }
    return [];
}

// Save candles to appropriate day file
function saveDayCandles(category, asset, dateStr, candles) {
    if (candles.length === 0) return;

    const date = new Date(dateStr);
    const weekStart = getWeekStart(date);
    const assetDir = path.join(DATA_DIR, category, asset);
    const weekFolder = path.join(assetDir, `week_${formatDate(weekStart)}`);
    ensureDir(weekFolder);

    const dayFile = path.join(weekFolder, `${dateStr}.json`);

    // Sort and deduplicate
    const unique = {};
    for (const c of candles) {
        unique[c.timestamp] = c;
    }
    const sorted = Object.values(unique).sort((a, b) => a.timestamp - b.timestamp);

    fs.writeFileSync(dayFile, JSON.stringify(sorted, null, 2));
    return sorted.length;
}

// Fetch historical data for an asset
function fetchHistoricalData(client, category, assetName, symbol) {
    return new Promise((resolve) => {
        const chart = new client.Session.Chart();

        // Request more historical data
        chart.setMarket(symbol, {
            timeframe: '1',
            range: 300, // Request 300 candles (~5 hours of 1-min data)
        });

        let resolved = false;
        const timeout = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                chart.delete();
                resolve({ success: false, error: 'timeout', candles: [] });
            }
        }, 15000);

        chart.onUpdate(() => {
            if (resolved) return;

            const periods = chart.periods;
            if (!periods || periods.length === 0) return;

            const candles = periods.map(p => ({
                timestamp: p.time * 1000,
                open: p.open,
                high: p.max,
                low: p.min,
                close: p.close,
                volume: p.volume || 0
            }));

            clearTimeout(timeout);
            resolved = true;
            chart.delete();
            resolve({ success: true, candles });
        });

        chart.onError((err) => {
            if (!resolved) {
                clearTimeout(timeout);
                resolved = true;
                chart.delete();
                resolve({ success: false, error: err, candles: [] });
            }
        });
    });
}

// Main backfill function
async function backfillAsset(client, category, assetName, symbol) {
    console.log(`\nBackfilling ${category}/${assetName}...`);

    const result = await fetchHistoricalData(client, category, assetName, symbol);

    if (!result.success || result.candles.length === 0) {
        console.log(`  Failed: ${result.error || 'no data'}`);
        return { success: false, filled: 0 };
    }

    console.log(`  Received ${result.candles.length} historical candles`);

    // Group candles by day
    const byDay = {};
    for (const candle of result.candles) {
        const dayStr = formatDate(new Date(candle.timestamp));
        if (!byDay[dayStr]) byDay[dayStr] = [];
        byDay[dayStr].push(candle);
    }

    let totalFilled = 0;

    // Merge with existing data for each day
    for (const [dayStr, newCandles] of Object.entries(byDay)) {
        const existing = loadDayCandles(category, assetName, dayStr);
        const existingTimestamps = new Set(existing.map(c => c.timestamp));

        // Find candles we don't have
        const toAdd = newCandles.filter(c => !existingTimestamps.has(c.timestamp));

        if (toAdd.length > 0) {
            const merged = [...existing, ...toAdd];
            const saved = saveDayCandles(category, assetName, dayStr, merged);
            console.log(`  ${dayStr}: Added ${toAdd.length} candles (total: ${saved})`);
            totalFilled += toAdd.length;
        }
    }

    if (totalFilled === 0) {
        console.log(`  No gaps to fill`);
    }

    return { success: true, filled: totalFilled };
}

async function main() {
    const args = process.argv.slice(2);
    const filterCategory = args[0];
    const filterAsset = args[1];

    console.log('============================================================');
    console.log('TradingView Data Backfill');
    console.log('============================================================');
    console.log(`Data directory: ${DATA_DIR}`);
    if (filterCategory) console.log(`Category filter: ${filterCategory}`);
    if (filterAsset) console.log(`Asset filter: ${filterAsset}`);
    console.log('============================================================\n');

    const client = new TradingView.Client();
    let totalFilled = 0;
    let assetsProcessed = 0;

    for (const [category, assets] of Object.entries(ASSETS)) {
        if (filterCategory && category !== filterCategory) continue;

        for (const [assetName, symbol] of Object.entries(assets)) {
            if (filterAsset && assetName !== filterAsset) continue;

            try {
                const result = await backfillAsset(client, category, assetName, symbol);
                if (result.success) {
                    totalFilled += result.filled;
                }
                assetsProcessed++;
            } catch (err) {
                console.log(`  Error: ${err.message}`);
            }

            // Small delay between assets
            await new Promise(r => setTimeout(r, 500));
        }
    }

    client.end();

    console.log('\n============================================================');
    console.log('Backfill Complete');
    console.log(`Assets processed: ${assetsProcessed}`);
    console.log(`Total candles filled: ${totalFilled}`);
    console.log('============================================================');
}

main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
