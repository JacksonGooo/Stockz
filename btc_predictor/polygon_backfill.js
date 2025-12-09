/**
 * Polygon.io Historical Data Backfill
 *
 * Fetches historical 1-minute candle data from Polygon.io API.
 * Free tier: 5 API calls/minute, 2 years of historical data
 *
 * Usage:
 *   node polygon_backfill.js                     - Backfill all assets (last 30 days)
 *   node polygon_backfill.js Crypto BTC          - Backfill specific asset
 *   node polygon_backfill.js Crypto BTC 90       - Backfill last 90 days
 *
 * Set your API key:
 *   set POLYGON_API_KEY=your_api_key_here
 *
 * Get a free API key at: https://polygon.io/
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// ============================================================
// CONFIGURATION
// ============================================================

// Get API key from environment variable or use default
const API_KEY = process.env.POLYGON_API_KEY || 'CgMLvEqWF_tjIwq4_o6CMsAXvJwmRjCw';

// Asset mappings - Polygon uses different symbols
// EXPANDED: Many more cryptos available on Polygon free tier!
const POLYGON_SYMBOLS = {
    Crypto: {
        // Top 15 (our main assets)
        BTC: 'X:BTCUSD',
        ETH: 'X:ETHUSD',
        SOL: 'X:SOLUSD',
        XRP: 'X:XRPUSD',
        DOGE: 'X:DOGEUSD',
        ADA: 'X:ADAUSD',
        AVAX: 'X:AVAXUSD',
        DOT: 'X:DOTUSD',
        POL: 'X:POLUSD',  // Polygon (formerly MATIC)
        LINK: 'X:LINKUSD',
        LTC: 'X:LTCUSD',
        UNI: 'X:UNIUSD',
        ATOM: 'X:ATOMUSD',
        XLM: 'X:XLMUSD',
        ALGO: 'X:ALGOUSD',
        // Additional popular cryptos
        BCH: 'X:BCHUSD',   // Bitcoin Cash
        ETC: 'X:ETCUSD',   // Ethereum Classic
        FIL: 'X:FILUSD',   // Filecoin
        AAVE: 'X:AAVEUSD', // Aave
        MKR: 'X:MKRUSD',   // Maker
        COMP: 'X:COMPUSD', // Compound
        SNX: 'X:SNXUSD',   // Synthetix
        SUSHI: 'X:SUSHIUSD', // SushiSwap
        YFI: 'X:YFIUSD',   // Yearn Finance
        CRV: 'X:CRVUSD',   // Curve DAO
        BAL: 'X:BALUSD',   // Balancer
        MATIC: 'X:MATICUSD', // Polygon/Matic
        SHIB: 'X:SHIBUSD', // Shiba Inu
        APE: 'X:APEUSD',   // ApeCoin
        ARB: 'X:ARBUSD',   // Arbitrum
        OP: 'X:OPUSD',     // Optimism
        NEAR: 'X:NEARUSD', // NEAR Protocol
        FTM: 'X:FTMUSD',   // Fantom
        SAND: 'X:SANDUSD', // The Sandbox
        MANA: 'X:MANAUSD', // Decentraland
        AXS: 'X:AXSUSD',   // Axie Infinity
        ENJ: 'X:ENJUSD',   // Enjin Coin
        GRT: 'X:GRTUSD',   // The Graph
        LRC: 'X:LRCUSD',   // Loopring
        IMX: 'X:IMXUSD',   // Immutable X
        HBAR: 'X:HBARUSD', // Hedera
        VET: 'X:VETUSD',   // VeChain
        EOS: 'X:EOSUSD',   // EOS
        TRX: 'X:TRXUSD',   // TRON
        XTZ: 'X:XTZUSD',   // Tezos
        THETA: 'X:THETAUSD', // Theta Network
        ZEC: 'X:ZECUSD',   // Zcash
        DASH: 'X:DASHUSD', // Dash
        NEO: 'X:NEOUSD',   // NEO
        XMR: 'X:XMRUSD',   // Monero
        WAVES: 'X:WAVESUSD', // Waves
        ZIL: 'X:ZILUSD',   // Zilliqa
        ONT: 'X:ONTUSD',   // Ontology
        ICX: 'X:ICXUSD',   // ICON
        QTUM: 'X:QTUMUSD', // Qtum
        ZRX: 'X:ZRXUSD',   // 0x Protocol
        BAT: 'X:BATUSD',   // Basic Attention Token
        KNC: 'X:KNCUSD',   // Kyber Network
        REN: 'X:RENUSD',   // Ren
        STORJ: 'X:STORJUSD', // Storj
        BNT: 'X:BNTUSD',   // Bancor
        OMG: 'X:OMGUSD',   // OMG Network
        ANKR: 'X:ANKRUSD', // Ankr
        SKL: 'X:SKLUSD',   // SKALE
        NMR: 'X:NMRUSD',   // Numeraire
        OGN: 'X:OGNUSD',   // Origin Protocol
        BAND: 'X:BANDUSD', // Band Protocol
    },
    "Stock Market": {
        // === ETFs - Major Index ===
        SPY: 'SPY',    // S&P 500
        QQQ: 'QQQ',    // Nasdaq 100
        DIA: 'DIA',    // Dow Jones
        IWM: 'IWM',    // Russell 2000
        VTI: 'VTI',    // Total Stock Market
        VOO: 'VOO',    // Vanguard S&P 500
        VXX: 'VXX',    // VIX Short-Term
        VIXY: 'VIXY', // VIX Short-Term Futures
        SQQQ: 'SQQQ', // 3x Inverse Nasdaq
        TQQQ: 'TQQQ', // 3x Long Nasdaq
        SPXS: 'SPXS', // 3x Inverse S&P 500
        SPXL: 'SPXL', // 3x Long S&P 500
        // === Sector ETFs ===
        XLK: 'XLK',   // Technology
        XLF: 'XLF',   // Financials
        XLE: 'XLE',   // Energy
        XLV: 'XLV',   // Healthcare
        XLI: 'XLI',   // Industrials
        XLY: 'XLY',   // Consumer Discretionary
        XLP: 'XLP',   // Consumer Staples
        XLB: 'XLB',   // Materials
        XLU: 'XLU',   // Utilities
        XLRE: 'XLRE', // Real Estate
        // === Tech Giants (FAANG+) ===
        AAPL: 'AAPL',
        MSFT: 'MSFT',
        GOOGL: 'GOOGL',
        GOOG: 'GOOG',
        AMZN: 'AMZN',
        NVDA: 'NVDA',
        META: 'META',
        TSLA: 'TSLA',
        // === Semiconductors ===
        AMD: 'AMD',
        INTC: 'INTC',
        AVGO: 'AVGO',  // Broadcom
        QCOM: 'QCOM',  // Qualcomm
        TXN: 'TXN',    // Texas Instruments
        MU: 'MU',      // Micron
        LRCX: 'LRCX',  // Lam Research
        AMAT: 'AMAT',  // Applied Materials
        KLAC: 'KLAC',  // KLA Corp
        MRVL: 'MRVL',  // Marvell
        ASML: 'ASML',  // ASML
        TSM: 'TSM',    // Taiwan Semiconductor
        // === Software & Cloud ===
        ORCL: 'ORCL',
        CRM: 'CRM',
        ADBE: 'ADBE',
        NOW: 'NOW',    // ServiceNow
        PANW: 'PANW',  // Palo Alto Networks
        CRWD: 'CRWD',  // CrowdStrike
        ZS: 'ZS',      // Zscaler
        DDOG: 'DDOG',  // Datadog
        SNOW: 'SNOW',  // Snowflake
        NET: 'NET',    // Cloudflare
        PLTR: 'PLTR',  // Palantir
        MDB: 'MDB',    // MongoDB
        TEAM: 'TEAM',  // Atlassian
        // === Finance ===
        JPM: 'JPM',
        V: 'V',
        MA: 'MA',
        BAC: 'BAC',
        GS: 'GS',
        MS: 'MS',
        C: 'C',
        WFC: 'WFC',
        BLK: 'BLK',    // BlackRock
        SCHW: 'SCHW',  // Charles Schwab
        AXP: 'AXP',    // American Express
        USB: 'USB',    // US Bank
        PNC: 'PNC',    // PNC Financial
        TFC: 'TFC',    // Truist
        // === Healthcare & Biotech ===
        UNH: 'UNH',
        JNJ: 'JNJ',
        PFE: 'PFE',
        ABBV: 'ABBV',
        MRK: 'MRK',
        LLY: 'LLY',
        TMO: 'TMO',    // Thermo Fisher
        ABT: 'ABT',    // Abbott
        BMY: 'BMY',    // Bristol-Myers
        AMGN: 'AMGN',  // Amgen
        GILD: 'GILD',  // Gilead
        MRNA: 'MRNA',  // Moderna
        BIIB: 'BIIB',  // Biogen
        REGN: 'REGN',  // Regeneron
        VRTX: 'VRTX',  // Vertex
        // === Consumer ===
        WMT: 'WMT',
        PG: 'PG',
        KO: 'KO',
        PEP: 'PEP',
        COST: 'COST',
        HD: 'HD',
        MCD: 'MCD',
        NKE: 'NKE',
        SBUX: 'SBUX',
        LOW: 'LOW',    // Lowe's
        TGT: 'TGT',    // Target
        DG: 'DG',      // Dollar General
        DLTR: 'DLTR',  // Dollar Tree
        TJX: 'TJX',    // TJX Companies
        ROST: 'ROST',  // Ross Stores
        CMG: 'CMG',    // Chipotle
        YUM: 'YUM',    // Yum! Brands
        DPZ: 'DPZ',    // Domino's
        // === Energy ===
        XOM: 'XOM',
        CVX: 'CVX',
        COP: 'COP',
        SLB: 'SLB',    // Schlumberger
        EOG: 'EOG',    // EOG Resources
        PXD: 'PXD',    // Pioneer Natural
        OXY: 'OXY',    // Occidental
        MPC: 'MPC',    // Marathon Petroleum
        VLO: 'VLO',    // Valero
        PSX: 'PSX',    // Phillips 66
        // === Tech & Internet ===
        NFLX: 'NFLX',
        PYPL: 'PYPL',
        SQ: 'SQ',
        COIN: 'COIN',
        UBER: 'UBER',
        ABNB: 'ABNB',
        LYFT: 'LYFT',
        DASH: 'DASH',  // DoorDash
        SHOP: 'SHOP',  // Shopify
        EBAY: 'EBAY',
        ETSY: 'ETSY',
        ZM: 'ZM',      // Zoom
        ROKU: 'ROKU',
        SPOT: 'SPOT',  // Spotify
        SNAP: 'SNAP',
        PINS: 'PINS',  // Pinterest
        TTD: 'TTD',    // The Trade Desk
        // === Industrial ===
        CAT: 'CAT',
        BA: 'BA',
        UPS: 'UPS',
        FDX: 'FDX',
        HON: 'HON',    // Honeywell
        GE: 'GE',      // GE Aerospace
        RTX: 'RTX',    // RTX Corp
        LMT: 'LMT',    // Lockheed Martin
        NOC: 'NOC',    // Northrop Grumman
        GD: 'GD',      // General Dynamics
        DE: 'DE',      // John Deere
        MMM: 'MMM',    // 3M
        // === Telecom & Media ===
        VZ: 'VZ',
        T: 'T',
        TMUS: 'TMUS',
        CMCSA: 'CMCSA', // Comcast
        CHTR: 'CHTR',   // Charter
        WBD: 'WBD',     // Warner Bros Discovery
        PARA: 'PARA',   // Paramount
        // === Auto ===
        GM: 'GM',
        F: 'F',        // Ford
        RIVN: 'RIVN',  // Rivian
        LCID: 'LCID',  // Lucid
        // === Meme Stocks & Retail Favorites ===
        GME: 'GME',    // GameStop
        AMC: 'AMC',    // AMC
        BBBY: 'BBBY',  // Bed Bath (if still trading)
        SOFI: 'SOFI',  // SoFi
        HOOD: 'HOOD',  // Robinhood
        AFRM: 'AFRM',  // Affirm
    },
    Commodities: {
        // Precious Metals (forex endpoint)
        GOLD: 'C:XAUUSD',     // Gold vs USD
        SILVER: 'C:XAGUSD',   // Silver vs USD
        PLATINUM: 'C:XPTUSD', // Platinum vs USD
        PALLADIUM: 'C:XPDUSD', // Palladium vs USD
        // Energy (ETFs as proxies - stocks endpoint)
        OIL: 'USO',           // United States Oil Fund ETF
        NATGAS: 'UNG',        // United States Natural Gas Fund ETF
        // Agriculture & Other Commodity ETFs
        CORN: 'CORN',         // Teucrium Corn Fund
        WHEAT: 'WEAT',        // Teucrium Wheat Fund
        SOYBEAN: 'SOYB',      // Teucrium Soybean Fund
        COPPER: 'CPER',       // United States Copper Index Fund
        // Broad Commodity ETFs
        DBC: 'DBC',           // Invesco DB Commodity Index
        GSG: 'GSG',           // iShares S&P GSCI Commodity
        PDBC: 'PDBC',         // Invesco Optimum Yield Diversified Commodity
    },
    Currencies: {
        // Major pairs
        EURUSD: 'C:EURUSD',
        GBPUSD: 'C:GBPUSD',
        USDJPY: 'C:USDJPY',
        AUDUSD: 'C:AUDUSD',
        USDCAD: 'C:USDCAD',
        USDCHF: 'C:USDCHF',
        NZDUSD: 'C:NZDUSD',
        // Cross pairs
        EURGBP: 'C:EURGBP',
        EURJPY: 'C:EURJPY',
        GBPJPY: 'C:GBPJPY',
        AUDJPY: 'C:AUDJPY',
        CADJPY: 'C:CADJPY',
        EURAUD: 'C:EURAUD',
        EURCHF: 'C:EURCHF',
        GBPCHF: 'C:GBPCHF',
        GBPAUD: 'C:GBPAUD',
        AUDNZD: 'C:AUDNZD',
        NZDJPY: 'C:NZDJPY',
        CHFJPY: 'C:CHFJPY',
        EURCAD: 'C:EURCAD',
        AUDCAD: 'C:AUDCAD',
        GBPCAD: 'C:GBPCAD',
        // Emerging market pairs
        USDMXN: 'C:USDMXN',   // Mexican Peso
        USDZAR: 'C:USDZAR',   // South African Rand
        USDTRY: 'C:USDTRY',   // Turkish Lira
        USDBRL: 'C:USDBRL',   // Brazilian Real
        USDINR: 'C:USDINR',   // Indian Rupee
        USDCNY: 'C:USDCNY',   // Chinese Yuan
        USDSGD: 'C:USDSGD',   // Singapore Dollar
        USDHKD: 'C:USDHKD',   // Hong Kong Dollar
        USDKRW: 'C:USDKRW',   // South Korean Won
        USDSEK: 'C:USDSEK',   // Swedish Krona
        USDNOK: 'C:USDNOK',   // Norwegian Krone
        USDDKK: 'C:USDDKK',   // Danish Krone
        USDPLN: 'C:USDPLN',   // Polish Zloty
        USDHUF: 'C:USDHUF',   // Hungarian Forint
        USDCZK: 'C:USDCZK',   // Czech Koruna
        USDRUB: 'C:USDRUB',   // Russian Ruble
        USDTHB: 'C:USDTHB',   // Thai Baht
        USDMYR: 'C:USDMYR',   // Malaysian Ringgit
        USDIDR: 'C:USDIDR',   // Indonesian Rupiah
        USDPHP: 'C:USDPHP',   // Philippine Peso
    },
};

const DATA_DIR = path.join(__dirname, '..', 'Data');
const RATE_LIMIT_DELAY = 12000; // 12 seconds (minimum for 5 calls/min free tier)

// ============================================================
// Helper Functions
// ============================================================

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

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================
// Polygon API Functions
// ============================================================

function fetchFromPolygon(url) {
    return new Promise((resolve, reject) => {
        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    resolve(json);
                } catch (e) {
                    reject(new Error(`Failed to parse response: ${data.substring(0, 200)}`));
                }
            });
        }).on('error', reject);
    });
}

/**
 * Fetch 1-minute candles from Polygon.io
 * @param {string} symbol - Polygon symbol (e.g., 'X:BTCUSD', 'AAPL', 'C:EURUSD')
 * @param {string} from - Start date (YYYY-MM-DD)
 * @param {string} to - End date (YYYY-MM-DD)
 */
async function fetchPolygonCandles(symbol, from, to) {
    // Determine if it's crypto, stock, or forex based on symbol prefix
    let endpoint;
    if (symbol.startsWith('X:')) {
        // Crypto
        endpoint = `https://api.polygon.io/v2/aggs/ticker/${symbol}/range/1/minute/${from}/${to}?adjusted=true&sort=asc&limit=50000&apiKey=${API_KEY}`;
    } else if (symbol.startsWith('C:')) {
        // Forex
        endpoint = `https://api.polygon.io/v2/aggs/ticker/${symbol}/range/1/minute/${from}/${to}?adjusted=true&sort=asc&limit=50000&apiKey=${API_KEY}`;
    } else {
        // Stocks
        endpoint = `https://api.polygon.io/v2/aggs/ticker/${symbol}/range/1/minute/${from}/${to}?adjusted=true&sort=asc&limit=50000&apiKey=${API_KEY}`;
    }

    console.log(`  Fetching ${symbol} from ${from} to ${to}...`);

    const response = await fetchFromPolygon(endpoint);

    if (response.status === 'ERROR') {
        throw new Error(response.error || 'Unknown API error');
    }

    if (!response.results || response.results.length === 0) {
        return [];
    }

    // Convert Polygon format to our format
    const candles = response.results.map(r => ({
        timestamp: r.t, // Already in milliseconds
        open: r.o,
        high: r.h,
        low: r.l,
        close: r.c,
        volume: r.v || 0,
    }));

    return candles;
}

// ============================================================
// Data Storage Functions
// ============================================================

function loadDayCandles(category, asset, dateStr) {
    const assetDir = path.join(DATA_DIR, category, asset);

    if (!fs.existsSync(assetDir)) {
        return [];
    }

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

function saveDayCandles(category, asset, dateStr, candles) {
    if (candles.length === 0) return 0;

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

// ============================================================
// Main Backfill Logic
// ============================================================

async function backfillAsset(category, assetName, polygonSymbol, daysBack) {
    console.log(`\nBackfilling ${category}/${assetName} (${polygonSymbol})...`);

    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - daysBack);

    let totalFetched = 0;
    let totalNew = 0;

    // Fetch in chunks (Polygon limits results per request)
    // For free tier, we need to be careful about rate limits
    let currentStart = new Date(startDate);

    while (currentStart < endDate) {
        // Fetch 1 day at a time to stay under limits
        const currentEnd = new Date(currentStart);
        currentEnd.setDate(currentEnd.getDate() + 1);

        if (currentEnd > endDate) {
            currentEnd.setTime(endDate.getTime());
        }

        const fromStr = formatDate(currentStart);
        const toStr = formatDate(currentEnd);

        try {
            const candles = await fetchPolygonCandles(polygonSymbol, fromStr, toStr);
            totalFetched += candles.length;

            if (candles.length > 0) {
                // Group by day and merge with existing
                const byDay = {};
                for (const candle of candles) {
                    const dayStr = formatDate(new Date(candle.timestamp));
                    if (!byDay[dayStr]) byDay[dayStr] = [];
                    byDay[dayStr].push(candle);
                }

                for (const [dayStr, dayCandles] of Object.entries(byDay)) {
                    const existing = loadDayCandles(category, assetName, dayStr);
                    const existingTimestamps = new Set(existing.map(c => c.timestamp));

                    const newCandles = dayCandles.filter(c => !existingTimestamps.has(c.timestamp));

                    if (newCandles.length > 0) {
                        const merged = [...existing, ...newCandles];
                        const saved = saveDayCandles(category, assetName, dayStr, merged);
                        console.log(`    ${dayStr}: +${newCandles.length} new candles (total: ${saved})`);
                        totalNew += newCandles.length;
                    }
                }
            }

            // Rate limiting for free tier
            await sleep(RATE_LIMIT_DELAY);

        } catch (err) {
            console.log(`    Error fetching ${fromStr}: ${err.message}`);
            // Continue with next day
        }

        currentStart.setDate(currentStart.getDate() + 1);
    }

    console.log(`  Fetched ${totalFetched} candles, ${totalNew} new`);
    return { fetched: totalFetched, new: totalNew };
}

async function main() {
    const args = process.argv.slice(2);
    const filterCategory = args[0];
    const filterAsset = args[1];
    const daysBack = parseInt(args[2]) || 30;

    console.log('============================================================');
    console.log('Polygon.io Historical Data Backfill');
    console.log('============================================================');

    if (!API_KEY) {
        console.log('\nERROR: No API key found!');
        console.log('');
        console.log('Set your Polygon.io API key:');
        console.log('  Windows: set POLYGON_API_KEY=your_api_key_here');
        console.log('  Linux/Mac: export POLYGON_API_KEY=your_api_key_here');
        console.log('');
        console.log('Get a free API key at: https://polygon.io/');
        console.log('');
        console.log('Free tier limits:');
        console.log('  - 5 API calls per minute');
        console.log('  - 2 years of historical data');
        console.log('  - End-of-day data only for stocks (1-min requires paid)');
        console.log('============================================================');
        process.exit(1);
    }

    console.log(`API Key: ${API_KEY.substring(0, 8)}...`);
    console.log(`Data directory: ${DATA_DIR}`);
    console.log(`Days to backfill: ${daysBack}`);
    if (filterCategory) console.log(`Category filter: ${filterCategory}`);
    if (filterAsset) console.log(`Asset filter: ${filterAsset}`);
    console.log('============================================================');
    console.log('');
    console.log('NOTE: Free Polygon.io tier has limitations:');
    console.log('  - 5 API calls per minute (15 sec delay between requests)');
    console.log('  - Crypto: Full 1-minute data available');
    console.log('  - Stocks: May require paid tier for 1-minute data');
    console.log('  - Forex: May require paid tier for 1-minute data');
    console.log('');
    console.log('This will take a while due to rate limits...');
    console.log('============================================================\n');

    let totalFetched = 0;
    let totalNew = 0;
    let assetsProcessed = 0;

    for (const [category, assets] of Object.entries(POLYGON_SYMBOLS)) {
        if (filterCategory && category !== filterCategory) continue;

        for (const [assetName, polygonSymbol] of Object.entries(assets)) {
            if (filterAsset && assetName !== filterAsset) continue;

            try {
                const result = await backfillAsset(category, assetName, polygonSymbol, daysBack);
                totalFetched += result.fetched;
                totalNew += result.new;
                assetsProcessed++;
            } catch (err) {
                console.log(`  Error: ${err.message}`);
            }
        }
    }

    console.log('\n============================================================');
    console.log('Backfill Complete');
    console.log(`Assets processed: ${assetsProcessed}`);
    console.log(`Total candles fetched: ${totalFetched}`);
    console.log(`Total new candles saved: ${totalNew}`);
    console.log('============================================================');
}

main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
