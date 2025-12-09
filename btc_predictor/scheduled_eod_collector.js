/**
 * Scheduled End-of-Day Data Collector
 * Runs once at 4:05 PM ET to collect the full day's stock data
 * Also collects crypto/forex/commodities data
 */

const TradingView = require('@nicepkg/tradingview-ws');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');

// All assets to collect
const ASSETS = {
    // Crypto - 24/7 markets
    crypto: {
        BTC: 'BINANCE:BTCUSDT', ETH: 'BINANCE:ETHUSDT', SOL: 'BINANCE:SOLUSDT',
        XRP: 'BINANCE:XRPUSDT', DOGE: 'BINANCE:DOGEUSDT', ADA: 'BINANCE:ADAUSDT',
        AVAX: 'BINANCE:AVAXUSDT', DOT: 'BINANCE:DOTUSDT', LINK: 'BINANCE:LINKUSDT',
        LTC: 'BINANCE:LTCUSDT', UNI: 'BINANCE:UNIUSDT', ATOM: 'BINANCE:ATOMUSDT',
        XLM: 'BINANCE:XLMUSDT', ALGO: 'BINANCE:ALGOUSDT', BCH: 'BINANCE:BCHUSDT',
        ETC: 'BINANCE:ETCUSDT', FIL: 'BINANCE:FILUSDT', AAVE: 'BINANCE:AAVEUSDT',
        MKR: 'BINANCE:MKRUSDT', COMP: 'BINANCE:COMPUSDT', SHIB: 'BINANCE:SHIBUSDT',
        ARB: 'BINANCE:ARBUSDT', NEAR: 'BINANCE:NEARUSDT', SAND: 'BINANCE:SANDUSDT',
        MANA: 'BINANCE:MANAUSDT', GRT: 'BINANCE:GRTUSDT', ICP: 'BINANCE:ICPUSDT',
        HBAR: 'BINANCE:HBARUSDT', QNT: 'COINBASE:QNTUSD', ZEC: 'BINANCE:ZECUSDT',
        DASH: 'KRAKEN:DASHUSD', ENS: 'BINANCE:ENSUSDT', ANKR: 'BINANCE:ANKRUSDT'
    },

    // Stock Market - closes 4 PM ET
    stocks: {
        // Major ETFs
        SPY: 'AMEX:SPY', QQQ: 'NASDAQ:QQQ', DIA: 'AMEX:DIA', IWM: 'AMEX:IWM',
        VTI: 'AMEX:VTI', VOO: 'AMEX:VOO',
        // Leveraged ETFs
        TQQQ: 'NASDAQ:TQQQ', SQQQ: 'NASDAQ:SQQQ', SPXL: 'AMEX:SPXL', SPXS: 'AMEX:SPXS',
        // Sector ETFs
        XLK: 'AMEX:XLK', XLF: 'AMEX:XLF', XLE: 'AMEX:XLE', XLV: 'AMEX:XLV',
        // Tech Giants
        AAPL: 'NASDAQ:AAPL', MSFT: 'NASDAQ:MSFT', GOOGL: 'NASDAQ:GOOGL', AMZN: 'NASDAQ:AMZN',
        NVDA: 'NASDAQ:NVDA', META: 'NASDAQ:META', TSLA: 'NASDAQ:TSLA',
        // Semiconductors
        AMD: 'NASDAQ:AMD', INTC: 'NASDAQ:INTC', AVGO: 'NASDAQ:AVGO', QCOM: 'NASDAQ:QCOM',
        MU: 'NASDAQ:MU', TSM: 'NYSE:TSM', ASML: 'NASDAQ:ASML',
        // Financials
        JPM: 'NYSE:JPM', BAC: 'NYSE:BAC', GS: 'NYSE:GS', V: 'NYSE:V', MA: 'NYSE:MA',
        // Healthcare
        UNH: 'NYSE:UNH', JNJ: 'NYSE:JNJ', PFE: 'NYSE:PFE', LLY: 'NYSE:LLY',
        // Consumer
        WMT: 'NYSE:WMT', COST: 'NASDAQ:COST', HD: 'NYSE:HD', MCD: 'NYSE:MCD',
        // Energy
        XOM: 'NYSE:XOM', CVX: 'NYSE:CVX', COP: 'NYSE:COP',
        // Bitcoin-Related Stocks
        COIN: 'NASDAQ:COIN', MSTR: 'NASDAQ:MSTR', MARA: 'NASDAQ:MARA', RIOT: 'NASDAQ:RIOT',
        CLSK: 'NASDAQ:CLSK', GBTC: 'AMEX:GBTC', BITO: 'AMEX:BITO', IBIT: 'NASDAQ:IBIT',
        // Other Tech/Growth
        NFLX: 'NASDAQ:NFLX', CRM: 'NYSE:CRM', ADBE: 'NASDAQ:ADBE', ORCL: 'NYSE:ORCL',
        PYPL: 'NASDAQ:PYPL', SQ: 'NYSE:SQ', SHOP: 'NYSE:SHOP', UBER: 'NYSE:UBER',
        // Meme/Retail
        GME: 'NYSE:GME', AMC: 'NYSE:AMC', HOOD: 'NASDAQ:HOOD', SOFI: 'NASDAQ:SOFI'
    },

    // Commodities
    commodities: {
        GOLD: 'TVC:GOLD', SILVER: 'TVC:SILVER', OIL: 'TVC:USOIL',
        NATGAS: 'TVC:NATGAS', COPPER: 'COMEX:HG1!'
    },

    // Major Forex pairs
    forex: {
        EURUSD: 'FX:EURUSD', GBPUSD: 'FX:GBPUSD', USDJPY: 'FX:USDJPY',
        AUDUSD: 'FX:AUDUSD', USDCAD: 'FX:USDCAD', USDCHF: 'FX:USDCHF'
    }
};

// Get candles for past N minutes
async function getHistoricalCandles(symbol, minutes = 500) {
    return new Promise((resolve) => {
        const candles = [];
        const tv = new TradingView.Client();

        const timeout = setTimeout(() => {
            tv.end();
            resolve(candles);
        }, 30000);

        tv.onError((...err) => {
            clearTimeout(timeout);
            tv.end();
            resolve(candles);
        });

        const chart = new tv.Session.Chart();
        chart.setMarket(symbol, { timeframe: '1', range: minutes });

        chart.onUpdate(() => {
            const periods = chart.periods;
            if (periods && periods.length > 0) {
                for (const p of periods) {
                    if (p.time && p.open && p.high && p.low && p.close) {
                        candles.push({
                            timestamp: p.time * 1000,
                            open: p.open,
                            high: p.high,
                            low: p.low,
                            close: p.close,
                            volume: p.volume || 0
                        });
                    }
                }
                clearTimeout(timeout);
                tv.end();
                resolve(candles);
            }
        });
    });
}

// Save candles to file
function saveCandles(category, symbol, candles) {
    const dir = path.join(DATA_DIR, category);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const filePath = path.join(dir, `${symbol}.json`);
    let existing = [];

    if (fs.existsSync(filePath)) {
        try {
            existing = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        } catch (e) {}
    }

    // Merge and dedupe by timestamp
    const merged = [...existing, ...candles];
    const unique = Array.from(new Map(merged.map(c => [c.timestamp, c])).values());
    unique.sort((a, b) => a.timestamp - b.timestamp);

    fs.writeFileSync(filePath, JSON.stringify(unique, null, 2));
    return unique.length - existing.length;
}

// Main collection function
async function collectAllData() {
    console.log('============================================================');
    console.log('End-of-Day Data Collection');
    console.log('Started:', new Date().toISOString());
    console.log('============================================================\n');

    const results = { success: 0, failed: 0, newCandles: 0 };

    for (const [category, assets] of Object.entries(ASSETS)) {
        console.log(`\n=== ${category.toUpperCase()} ===\n`);

        for (const [symbol, tvSymbol] of Object.entries(assets)) {
            try {
                // Get 500 candles (~8 hours for stocks, covers full day)
                const candles = await getHistoricalCandles(tvSymbol, 500);

                if (candles.length > 0) {
                    const newCount = saveCandles(category === 'stocks' ? 'Stock Market' :
                                                 category === 'crypto' ? 'Crypto' :
                                                 category === 'forex' ? 'Currencies' : 'Commodities',
                                                 symbol, candles);
                    console.log(`  ${symbol}: ${candles.length} candles fetched, ${newCount} new`);
                    results.success++;
                    results.newCandles += newCount;
                } else {
                    console.log(`  ${symbol}: No data received`);
                    results.failed++;
                }
            } catch (err) {
                console.log(`  ${symbol}: Error - ${err.message}`);
                results.failed++;
            }

            // Small delay between requests
            await new Promise(r => setTimeout(r, 500));
        }
    }

    console.log('\n============================================================');
    console.log('Collection Complete');
    console.log(`Success: ${results.success} | Failed: ${results.failed}`);
    console.log(`New candles added: ${results.newCandles}`);
    console.log('Finished:', new Date().toISOString());
    console.log('============================================================');

    return results;
}

// Check if we should run (after 4 PM ET on weekdays)
function shouldRunNow() {
    const now = new Date();
    const etHour = parseInt(now.toLocaleString('en-US', { timeZone: 'America/New_York', hour: 'numeric', hour12: false }));
    const day = now.getDay();

    // Run if it's a weekday and after 4 PM ET
    return day >= 1 && day <= 5 && etHour >= 16;
}

// Run with optional scheduling
async function main() {
    const args = process.argv.slice(2);

    if (args.includes('--now')) {
        // Run immediately
        await collectAllData();
    } else if (args.includes('--schedule')) {
        // Wait until 4:05 PM ET then run
        console.log('Waiting for 4:05 PM ET to collect data...');

        const checkInterval = setInterval(async () => {
            const now = new Date();
            const etTime = now.toLocaleString('en-US', {
                timeZone: 'America/New_York',
                hour: 'numeric',
                minute: 'numeric',
                hour12: false
            });
            const [hour, minute] = etTime.split(':').map(Number);

            if (hour === 16 && minute >= 5 && minute < 10) {
                clearInterval(checkInterval);
                await collectAllData();
                process.exit(0);
            }
        }, 60000); // Check every minute
    } else {
        console.log('Usage:');
        console.log('  node scheduled_eod_collector.js --now      Run collection immediately');
        console.log('  node scheduled_eod_collector.js --schedule Wait for 4:05 PM ET then run');
    }
}

main().catch(console.error);
