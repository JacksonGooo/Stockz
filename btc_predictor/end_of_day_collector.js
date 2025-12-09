/**
 * End-of-Day Data Collector - TradingView
 *
 * Collects data throughout the day in memory, then saves to disk at end of day.
 * Keeps a rolling 60-minute window in memory for real-time predictions.
 *
 * Benefits:
 * - Less disk I/O during the day
 * - Complete daily files (no partial data)
 * - Always has recent data available for predictions
 */

const TradingView = require('@mathieuc/tradingview');
const fs = require('fs');
const path = require('path');

// ============================================================
// CONFIGURATION - ALL AVAILABLE ASSETS (EXPANDED - 200+ assets)
// ============================================================
const ASSETS = {
    Crypto: {
        // === Major Cryptos ===
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
        // === DeFi ===
        BCH: 'COINBASE:BCHUSD',
        ETC: 'COINBASE:ETCUSD',
        FIL: 'COINBASE:FILUSD',
        AAVE: 'COINBASE:AAVEUSD',
        MKR: 'COINBASE:MKRUSD',
        COMP: 'COINBASE:COMPUSD',
        SNX: 'COINBASE:SNXUSD',
        SUSHI: 'COINBASE:SUSHIUSD',
        YFI: 'COINBASE:YFIUSD',
        CRV: 'COINBASE:CRVUSD',
        BAL: 'COINBASE:BALUSD',
        // === Layer 2 & New ===
        SHIB: 'COINBASE:SHIBUSD',
        APE: 'COINBASE:APEUSD',
        ARB: 'COINBASE:ARBUSD',
        OP: 'COINBASE:OPUSD',
        NEAR: 'COINBASE:NEARUSD',
        FTM: 'COINBASE:FTMUSD',
        // === Metaverse & Gaming ===
        SAND: 'COINBASE:SANDUSD',
        MANA: 'COINBASE:MANAUSD',
        AXS: 'COINBASE:AXSUSD',
        ENJ: 'COINBASE:ENJUSD',
        GRT: 'COINBASE:GRTUSD',
        LRC: 'COINBASE:LRCUSD',
        // === More Altcoins ===
        ICP: 'COINBASE:ICPUSD',
        VET: 'BINANCE:VETUSD',
        HBAR: 'COINBASE:HBARUSD',
        QNT: 'COINBASE:QNTUSD',
        EGLD: 'BINANCE:EGLDUSD',
        THETA: 'BINANCE:THETAUSD',
        XTZ: 'COINBASE:XTZUSD',
        EOS: 'COINBASE:EOSUSD',
        FLOW: 'COINBASE:FLOWUSD',
        CHZ: 'COINBASE:CHZUSD',
        KAVA: 'COINBASE:KAVAUSD',
        ROSE: 'COINBASE:ROSEUSD',
        ZEC: 'COINBASE:ZECUSD',
        DASH: 'COINBASE:DASHUSD',
        NEO: 'BINANCE:NEOUSD',
        WAVES: 'BINANCE:WAVESUSD',
        ZIL: 'BINANCE:ZILUSD',
        ENS: 'COINBASE:ENSUSD',
        DYDX: 'COINBASE:DYDXUSD',
        ONE: 'BINANCE:ONEUSD',
        CELO: 'COINBASE:CELOUSD',
        ANKR: 'COINBASE:ANKRUSD',
    },
    "Stock Market": {
        // === ETFs - Major Index ===
        SPY: 'AMEX:SPY',
        QQQ: 'NASDAQ:QQQ',
        DIA: 'AMEX:DIA',
        IWM: 'AMEX:IWM',
        VTI: 'AMEX:VTI',
        VOO: 'AMEX:VOO',
        VXX: 'CBOE:VXX',
        VIXY: 'AMEX:VIXY',
        SQQQ: 'NASDAQ:SQQQ',
        TQQQ: 'NASDAQ:TQQQ',
        SPXS: 'AMEX:SPXS',
        SPXL: 'AMEX:SPXL',
        // === Sector ETFs ===
        XLK: 'AMEX:XLK',
        XLF: 'AMEX:XLF',
        XLE: 'AMEX:XLE',
        XLV: 'AMEX:XLV',
        XLI: 'AMEX:XLI',
        XLY: 'AMEX:XLY',
        XLP: 'AMEX:XLP',
        XLB: 'AMEX:XLB',
        XLU: 'AMEX:XLU',
        XLRE: 'AMEX:XLRE',
        // === Tech Giants (FAANG+) ===
        AAPL: 'NASDAQ:AAPL',
        MSFT: 'NASDAQ:MSFT',
        GOOGL: 'NASDAQ:GOOGL',
        GOOG: 'NASDAQ:GOOG',
        AMZN: 'NASDAQ:AMZN',
        NVDA: 'NASDAQ:NVDA',
        META: 'NASDAQ:META',
        TSLA: 'NASDAQ:TSLA',
        // === Semiconductors ===
        AMD: 'NASDAQ:AMD',
        INTC: 'NASDAQ:INTC',
        AVGO: 'NASDAQ:AVGO',
        QCOM: 'NASDAQ:QCOM',
        TXN: 'NASDAQ:TXN',
        MU: 'NASDAQ:MU',
        LRCX: 'NASDAQ:LRCX',
        AMAT: 'NASDAQ:AMAT',
        KLAC: 'NASDAQ:KLAC',
        MRVL: 'NASDAQ:MRVL',
        ASML: 'NASDAQ:ASML',
        TSM: 'NYSE:TSM',
        // === Software & Cloud ===
        ORCL: 'NYSE:ORCL',
        CRM: 'NYSE:CRM',
        ADBE: 'NASDAQ:ADBE',
        NOW: 'NYSE:NOW',
        PANW: 'NASDAQ:PANW',
        CRWD: 'NASDAQ:CRWD',
        ZS: 'NASDAQ:ZS',
        DDOG: 'NASDAQ:DDOG',
        SNOW: 'NYSE:SNOW',
        NET: 'NYSE:NET',
        PLTR: 'NYSE:PLTR',
        MDB: 'NASDAQ:MDB',
        TEAM: 'NASDAQ:TEAM',
        // === Finance ===
        JPM: 'NYSE:JPM',
        V: 'NYSE:V',
        MA: 'NYSE:MA',
        BAC: 'NYSE:BAC',
        GS: 'NYSE:GS',
        MS: 'NYSE:MS',
        C: 'NYSE:C',
        WFC: 'NYSE:WFC',
        BLK: 'NYSE:BLK',
        SCHW: 'NYSE:SCHW',
        AXP: 'NYSE:AXP',
        USB: 'NYSE:USB',
        PNC: 'NYSE:PNC',
        TFC: 'NYSE:TFC',
        // === Healthcare & Biotech ===
        UNH: 'NYSE:UNH',
        JNJ: 'NYSE:JNJ',
        PFE: 'NYSE:PFE',
        ABBV: 'NYSE:ABBV',
        MRK: 'NYSE:MRK',
        LLY: 'NYSE:LLY',
        TMO: 'NYSE:TMO',
        ABT: 'NYSE:ABT',
        BMY: 'NYSE:BMY',
        AMGN: 'NASDAQ:AMGN',
        GILD: 'NASDAQ:GILD',
        MRNA: 'NASDAQ:MRNA',
        BIIB: 'NASDAQ:BIIB',
        REGN: 'NASDAQ:REGN',
        VRTX: 'NASDAQ:VRTX',
        // === Consumer ===
        WMT: 'NYSE:WMT',
        PG: 'NYSE:PG',
        KO: 'NYSE:KO',
        PEP: 'NASDAQ:PEP',
        COST: 'NASDAQ:COST',
        HD: 'NYSE:HD',
        MCD: 'NYSE:MCD',
        NKE: 'NYSE:NKE',
        SBUX: 'NASDAQ:SBUX',
        LOW: 'NYSE:LOW',
        TGT: 'NYSE:TGT',
        DG: 'NYSE:DG',
        DLTR: 'NASDAQ:DLTR',
        TJX: 'NYSE:TJX',
        ROST: 'NASDAQ:ROST',
        CMG: 'NYSE:CMG',
        YUM: 'NYSE:YUM',
        DPZ: 'NYSE:DPZ',
        // === Energy ===
        XOM: 'NYSE:XOM',
        CVX: 'NYSE:CVX',
        COP: 'NYSE:COP',
        SLB: 'NYSE:SLB',
        EOG: 'NYSE:EOG',
        PXD: 'NYSE:PXD',
        OXY: 'NYSE:OXY',
        MPC: 'NYSE:MPC',
        VLO: 'NYSE:VLO',
        PSX: 'NYSE:PSX',
        // === Tech & Internet ===
        NFLX: 'NASDAQ:NFLX',
        PYPL: 'NASDAQ:PYPL',
        SQ: 'NYSE:SQ',
        COIN: 'NASDAQ:COIN',
        // === Bitcoin-Related Stocks ===
        MSTR: 'NASDAQ:MSTR',   // MicroStrategy - largest corporate BTC holder
        GBTC: 'AMEX:GBTC',     // Grayscale Bitcoin Trust
        BITO: 'AMEX:BITO',     // ProShares Bitcoin Strategy ETF
        IBIT: 'NASDAQ:IBIT',   // iShares Bitcoin Trust (BlackRock)
        MARA: 'NASDAQ:MARA',   // Marathon Digital Holdings (BTC mining)
        RIOT: 'NASDAQ:RIOT',   // Riot Platforms (BTC mining)
        CLSK: 'NASDAQ:CLSK',   // CleanSpark (BTC mining)
        UBER: 'NYSE:UBER',
        ABNB: 'NASDAQ:ABNB',
        LYFT: 'NASDAQ:LYFT',
        DASH: 'NYSE:DASH',
        SHOP: 'NYSE:SHOP',
        EBAY: 'NASDAQ:EBAY',
        ETSY: 'NASDAQ:ETSY',
        ZM: 'NASDAQ:ZM',
        ROKU: 'NASDAQ:ROKU',
        SPOT: 'NYSE:SPOT',
        SNAP: 'NYSE:SNAP',
        PINS: 'NYSE:PINS',
        TTD: 'NASDAQ:TTD',
        // === Industrial ===
        CAT: 'NYSE:CAT',
        BA: 'NYSE:BA',
        UPS: 'NYSE:UPS',
        FDX: 'NYSE:FDX',
        HON: 'NASDAQ:HON',
        GE: 'NYSE:GE',
        RTX: 'NYSE:RTX',
        LMT: 'NYSE:LMT',
        NOC: 'NYSE:NOC',
        GD: 'NYSE:GD',
        DE: 'NYSE:DE',
        MMM: 'NYSE:MMM',
        // === Telecom & Media ===
        VZ: 'NYSE:VZ',
        T: 'NYSE:T',
        TMUS: 'NASDAQ:TMUS',
        CMCSA: 'NASDAQ:CMCSA',
        CHTR: 'NASDAQ:CHTR',
        WBD: 'NASDAQ:WBD',
        PARA: 'NASDAQ:PARA',
        // === Auto ===
        GM: 'NYSE:GM',
        F: 'NYSE:F',
        RIVN: 'NASDAQ:RIVN',
        LCID: 'NASDAQ:LCID',
        // === Meme Stocks & Retail Favorites ===
        GME: 'NYSE:GME',
        AMC: 'NYSE:AMC',
        SOFI: 'NASDAQ:SOFI',
        HOOD: 'NASDAQ:HOOD',
        AFRM: 'NASDAQ:AFRM',
    },
    Commodities: {
        // Precious Metals
        GOLD: 'TVC:GOLD',
        SILVER: 'TVC:SILVER',
        PLATINUM: 'TVC:PLATINUM',
        PALLADIUM: 'TVC:PALLADIUM',
        // Energy
        OIL: 'TVC:USOIL',
        BRENT: 'TVC:UKOIL',
        NATGAS: 'NYMEX:NG1!',
        RBOB: 'NYMEX:RB1!',
        HEATING_OIL: 'NYMEX:HO1!',
        // Metals & Materials
        COPPER: 'COMEX:HG1!',
        ALUMINUM: 'COMEX:ALI1!',
        ZINC: 'COMEX:ZINC1!',
        NICKEL: 'COMEX:NI1!',
        // Agriculture - Grains
        CORN_F: 'CBOT:ZC1!',
        WHEAT_F: 'CBOT:ZW1!',
        SOYBEAN_F: 'CBOT:ZS1!',
        OATS: 'CBOT:ZO1!',
        RICE: 'CBOT:ZR1!',
        // Agriculture - Softs
        COFFEE: 'ICEUS:KC1!',
        COCOA: 'ICEUS:CC1!',
        SUGAR: 'ICEUS:SB1!',
        COTTON: 'ICEUS:CT1!',
        OJ: 'ICEUS:OJ1!',
        // Livestock
        CATTLE: 'CME:LE1!',
        HOGS: 'CME:HE1!',
        FEEDER: 'CME:GF1!',
        // Commodity ETFs
        CORN: 'AMEX:CORN',
        WHEAT: 'AMEX:WEAT',
        SOYBEAN: 'AMEX:SOYB',
        DBC: 'AMEX:DBC',
        GSG: 'AMEX:GSG',
        GLD: 'AMEX:GLD',
        SLV: 'AMEX:SLV',
        USO: 'AMEX:USO',
        UNG: 'AMEX:UNG',
        PPLT: 'AMEX:PPLT',
        PALL: 'AMEX:PALL',
        CPER: 'AMEX:CPER',
        JJC: 'AMEX:JJC',
        DBA: 'AMEX:DBA',
        DJP: 'AMEX:DJP',
    },
    Indices: {
        // US Indices
        SP500: 'SP:SPX',
        NASDAQ: 'NASDAQ:IXIC',
        DOW: 'DJ:DJI',
        RUSSELL: 'TVC:RUT',
        VIX: 'CBOE:VIX',
        // European Indices
        DAX: 'XETR:DAX',
        FTSE: 'TVC:UKX',
        CAC40: 'EURONEXT:PX1',
        STOXX50: 'EUREX:FESX1!',
        IBEX: 'BME:IBC',
        FTSEMIB: 'MIL:FTSEMIB',
        SMI: 'SIX:SMI',
        AEX: 'EURONEXT:AEX',
        // Asian Indices
        NIKKEI: 'TVC:NI225',
        HANGSENG: 'TVC:HSI',
        SHANGHAI: 'SSE:000001',
        KOSPI: 'KRX:KOSPI',
        ASX200: 'ASX:XJO',
        NIFTY: 'NSE:NIFTY',
        SENSEX: 'BSE:SENSEX',
        TAIWAN: 'TWSE:TAIEX',
        // Other Indices
        TSX: 'TSX:TSX',
        BOVESPA: 'BMFBOVESPA:IBOV',
        MERVAL: 'BCBA:IMV',
    },
    Currencies: {
        // Major pairs
        EURUSD: 'FX:EURUSD',
        GBPUSD: 'FX:GBPUSD',
        USDJPY: 'FX:USDJPY',
        AUDUSD: 'FX:AUDUSD',
        USDCAD: 'FX:USDCAD',
        USDCHF: 'FX:USDCHF',
        NZDUSD: 'FX:NZDUSD',
        // Cross pairs
        EURGBP: 'FX:EURGBP',
        EURJPY: 'FX:EURJPY',
        GBPJPY: 'FX:GBPJPY',
        AUDJPY: 'FX:AUDJPY',
        CADJPY: 'FX:CADJPY',
        EURAUD: 'FX:EURAUD',
        EURCHF: 'FX:EURCHF',
        GBPCHF: 'FX:GBPCHF',
        GBPAUD: 'FX:GBPAUD',
        AUDNZD: 'FX:AUDNZD',
        NZDJPY: 'FX:NZDJPY',
        CHFJPY: 'FX:CHFJPY',
        EURCAD: 'FX:EURCAD',
        AUDCAD: 'FX:AUDCAD',
        GBPCAD: 'FX:GBPCAD',
        EURNZD: 'FX:EURNZD',
        GBPNZD: 'FX:GBPNZD',
        AUDCHF: 'FX:AUDCHF',
        CADCHF: 'FX:CADCHF',
        NZDCAD: 'FX:NZDCAD',
        NZDCHF: 'FX:NZDCHF',
        // Emerging market pairs
        USDMXN: 'FX:USDMXN',
        USDZAR: 'FX:USDZAR',
        USDTRY: 'FX:USDTRY',
        USDSEK: 'FX:USDSEK',
        USDNOK: 'FX:USDNOK',
        USDHKD: 'FX:USDHKD',
        USDSGD: 'FX:USDSGD',
        USDHUF: 'FX:USDHUF',
        USDPLN: 'FX:USDPLN',
        USDCZK: 'FX:USDCZK',
        USDRUB: 'FX:USDRUB',
        EURTRY: 'FX:EURTRY',
        EURMXN: 'FX:EURMXN',
        EURPLN: 'FX:EURPLN',
        EURHUF: 'FX:EURHUF',
        EURCZK: 'FX:EURCZK',
        EURSEK: 'FX:EURSEK',
        EURNOK: 'FX:EURNOK',
        // Dollar Index
        DXY: 'TVC:DXY',
    },
    Bonds: {
        // US Treasury
        US10Y: 'TVC:US10Y',
        US02Y: 'TVC:US02Y',
        US05Y: 'TVC:US05Y',
        US30Y: 'TVC:US30Y',
        US03M: 'TVC:US03M',
        // Other Government Bonds
        DE10Y: 'TVC:DE10Y',
        GB10Y: 'TVC:GB10Y',
        JP10Y: 'TVC:JP10Y',
        FR10Y: 'TVC:FR10Y',
        IT10Y: 'TVC:IT10Y',
        AU10Y: 'TVC:AU10Y',
        CN10Y: 'TVC:CN10Y',
        // Bond ETFs
        TLT: 'NASDAQ:TLT',
        IEF: 'NASDAQ:IEF',
        SHY: 'NASDAQ:SHY',
        BND: 'NASDAQ:BND',
        AGG: 'AMEX:AGG',
        LQD: 'AMEX:LQD',
        HYG: 'AMEX:HYG',
        JNK: 'AMEX:JNK',
        EMB: 'AMEX:EMB',
        TIPS: 'AMEX:TIP',
    },
    ETFs: {
        // International ETFs
        EFA: 'AMEX:EFA',
        EEM: 'AMEX:EEM',
        VEA: 'AMEX:VEA',
        VWO: 'AMEX:VWO',
        IEMG: 'AMEX:IEMG',
        // Country ETFs
        EWJ: 'AMEX:EWJ',
        FXI: 'AMEX:FXI',
        EWZ: 'AMEX:EWZ',
        EWG: 'AMEX:EWG',
        EWU: 'AMEX:EWU',
        EWC: 'AMEX:EWC',
        EWA: 'AMEX:EWA',
        EWY: 'AMEX:EWY',
        EWT: 'AMEX:EWT',
        EWH: 'AMEX:EWH',
        EWS: 'AMEX:EWS',
        INDA: 'AMEX:INDA',
        RSX: 'AMEX:RSX',
        // Sector ETFs
        XBI: 'AMEX:XBI',
        XHB: 'AMEX:XHB',
        XME: 'AMEX:XME',
        XOP: 'AMEX:XOP',
        XRT: 'AMEX:XRT',
        KRE: 'AMEX:KRE',
        KBE: 'AMEX:KBE',
        SMH: 'NASDAQ:SMH',
        IBB: 'NASDAQ:IBB',
        IYR: 'AMEX:IYR',
        ITA: 'AMEX:ITA',
        // Thematic ETFs
        ARKK: 'AMEX:ARKK',
        ARKG: 'AMEX:ARKG',
        ARKF: 'AMEX:ARKF',
        ARKW: 'AMEX:ARKW',
        BOTZ: 'NASDAQ:BOTZ',
        HACK: 'AMEX:HACK',
        SKYY: 'NASDAQ:SKYY',
        FINX: 'NASDAQ:FINX',
        ESPO: 'AMEX:ESPO',
        TAN: 'AMEX:TAN',
        ICLN: 'NASDAQ:ICLN',
        LIT: 'AMEX:LIT',
        // Leveraged ETFs
        UPRO: 'AMEX:UPRO',
        SPXU: 'AMEX:SPXU',
        QLD: 'NASDAQ:QLD',
        QID: 'NASDAQ:QID',
        UDOW: 'AMEX:UDOW',
        SDOW: 'AMEX:SDOW',
        TNA: 'AMEX:TNA',
        TZA: 'AMEX:TZA',
        LABU: 'AMEX:LABU',
        LABD: 'AMEX:LABD',
        SOXL: 'AMEX:SOXL',
        SOXS: 'AMEX:SOXS',
        NUGT: 'AMEX:NUGT',
        DUST: 'AMEX:DUST',
        JNUG: 'AMEX:JNUG',
        JDST: 'AMEX:JDST',
        UVXY: 'AMEX:UVXY',
        SVXY: 'AMEX:SVXY',
        // Dividend ETFs
        VIG: 'AMEX:VIG',
        VYM: 'AMEX:VYM',
        SCHD: 'AMEX:SCHD',
        DVY: 'AMEX:DVY',
        HDV: 'AMEX:HDV',
        SDY: 'AMEX:SDY',
        // REIT ETFs
        VNQ: 'AMEX:VNQ',
        VNQI: 'NASDAQ:VNQI',
        REM: 'AMEX:REM',
        // Factor ETFs
        MTUM: 'AMEX:MTUM',
        QUAL: 'AMEX:QUAL',
        VLUE: 'AMEX:VLUE',
        SIZE: 'AMEX:SIZE',
        USMV: 'AMEX:USMV',
    },
};

const DATA_DIR = path.join(__dirname, '..', 'Data');
const BUFFER_DIR = path.join(DATA_DIR, '.buffer'); // Buffer for realtime API access
const POLL_INTERVAL = 60000; // Poll every 60 seconds
const RECENT_CANDLES_COUNT = 60; // Keep last 60 minutes for predictions

// ============================================================
// In-Memory Buffer
// ============================================================
const memoryBuffer = {
    // Structure: { 'Category/Asset': { dayCandles: [], recentCandles: [], lastPrice: 0 } }
};

// Initialize buffer for all assets
function initializeBuffer() {
    for (const [category, assets] of Object.entries(ASSETS)) {
        for (const assetName of Object.keys(assets)) {
            const key = `${category}/${assetName}`;
            memoryBuffer[key] = {
                dayCandles: [],      // All candles for today
                recentCandles: [],   // Last 60 minutes for predictions
                lastPrice: 0,
                lastTimestamp: 0,
            };
        }
    }
}

// ============================================================
// Helper Functions
// ============================================================
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

function getCurrentDayKey() {
    return formatDate(new Date());
}

// Save all buffered candles for a day to disk
function saveDayToDisk(category, asset, dayKey, candles) {
    if (candles.length === 0) return;

    const date = new Date(dayKey);
    const weekStart = getWeekStart(date);
    const assetDir = path.join(DATA_DIR, category, asset);
    const weekFolder = path.join(assetDir, `week_${formatDate(weekStart)}`);
    ensureDir(weekFolder);

    const dayFile = path.join(weekFolder, `${dayKey}.json`);

    // Sort candles by timestamp
    candles.sort((a, b) => a.timestamp - b.timestamp);

    // Remove duplicates
    const uniqueCandles = [];
    const seen = new Set();
    for (const candle of candles) {
        if (!seen.has(candle.timestamp)) {
            seen.add(candle.timestamp);
            uniqueCandles.push(candle);
        }
    }

    fs.writeFileSync(dayFile, JSON.stringify(uniqueCandles, null, 2));
    console.log(`  Saved ${uniqueCandles.length} candles to ${dayFile}`);
}

// Add candle to memory buffer
function addCandleToBuffer(category, asset, candle) {
    const key = `${category}/${asset}`;
    const buffer = memoryBuffer[key];

    if (!buffer) return false;

    // Check if we already have this candle
    if (candle.timestamp <= buffer.lastTimestamp) {
        return false;
    }

    // Add to day candles
    buffer.dayCandles.push(candle);

    // Update recent candles (rolling window)
    buffer.recentCandles.push(candle);
    if (buffer.recentCandles.length > RECENT_CANDLES_COUNT) {
        buffer.recentCandles.shift();
    }

    buffer.lastPrice = candle.close;
    buffer.lastTimestamp = candle.timestamp;

    return true;
}

// Get recent candles for predictions
function getRecentCandles(category, asset) {
    const key = `${category}/${asset}`;
    return memoryBuffer[key]?.recentCandles || [];
}

// Write buffer to disk for API access
function writeBufferToDisk() {
    ensureDir(BUFFER_DIR);

    const bufferData = {};
    for (const [key, buffer] of Object.entries(memoryBuffer)) {
        bufferData[key] = {
            recentCandles: buffer.recentCandles,
            lastPrice: buffer.lastPrice,
            lastTimestamp: buffer.lastTimestamp,
            dayCount: buffer.dayCandles.length,
        };
    }

    const bufferFile = path.join(BUFFER_DIR, 'realtime.json');
    fs.writeFileSync(bufferFile, JSON.stringify(bufferData, null, 2));
}

// Export for API use
module.exports = {
    memoryBuffer,
    getRecentCandles,
    BUFFER_DIR,
};

// ============================================================
// Fetch data for a single asset
// ============================================================
function fetchAssetData(client, category, assetName, symbol) {
    return new Promise((resolve, reject) => {
        const chart = new client.Session.Chart();
        chart.setMarket(symbol, { timeframe: '1' });

        let resolved = false;
        const timeout = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                chart.delete();
                resolve({ success: false, error: 'timeout' });
            }
        }, 10000);

        chart.onUpdate(() => {
            if (resolved) return;

            const periods = chart.periods;
            if (!periods || periods.length === 0) return;

            let newCandles = 0;
            let lastPrice = 0;

            for (const period of periods) {
                const candle = {
                    timestamp: period.time * 1000,
                    open: period.open,
                    high: period.max,
                    low: period.min,
                    close: period.close,
                    volume: period.volume || 0
                };

                if (addCandleToBuffer(category, assetName, candle)) {
                    newCandles++;
                }
                lastPrice = candle.close;
            }

            clearTimeout(timeout);
            resolved = true;
            chart.delete();
            resolve({
                success: true,
                newCandles,
                lastPrice,
                totalPeriods: periods.length
            });
        });

        chart.onError((err) => {
            if (!resolved) {
                clearTimeout(timeout);
                resolved = true;
                chart.delete();
                resolve({ success: false, error: err });
            }
        });
    });
}

// ============================================================
// Poll all assets
// ============================================================
async function pollAllAssets() {
    const client = new TradingView.Client();
    const results = { success: 0, failed: 0, newCandles: 0 };

    console.log(`\n[${new Date().toISOString()}] Polling all assets...`);

    for (const [category, assets] of Object.entries(ASSETS)) {
        for (const [assetName, symbol] of Object.entries(assets)) {
            try {
                const result = await fetchAssetData(client, category, assetName, symbol);

                if (result.success) {
                    results.success++;
                    results.newCandles += result.newCandles;

                    if (result.newCandles > 0) {
                        console.log(`  ${category}/${assetName}: ${result.newCandles} new, last $${result.lastPrice.toFixed(2)}`);
                    }
                } else {
                    results.failed++;
                    if (result.error !== 'timeout') {
                        console.log(`  ${category}/${assetName}: Error - ${result.error}`);
                    }
                }
            } catch (err) {
                results.failed++;
                console.log(`  ${category}/${assetName}: Exception - ${err.message}`);
            }

            await new Promise(r => setTimeout(r, 100));
        }
    }

    client.end();

    // Write buffer to disk for API access
    writeBufferToDisk();

    console.log(`Poll complete: ${results.success} success, ${results.failed} failed, ${results.newCandles} new candles`);
    return results;
}

// ============================================================
// End-of-day save
// ============================================================
function saveAllToDisk() {
    const dayKey = getCurrentDayKey();
    console.log(`\n[${new Date().toISOString()}] Saving all data to disk for ${dayKey}...`);

    let totalSaved = 0;

    for (const [category, assets] of Object.entries(ASSETS)) {
        for (const assetName of Object.keys(assets)) {
            const key = `${category}/${assetName}`;
            const buffer = memoryBuffer[key];

            if (buffer && buffer.dayCandles.length > 0) {
                saveDayToDisk(category, assetName, dayKey, buffer.dayCandles);
                totalSaved += buffer.dayCandles.length;
            }
        }
    }

    console.log(`Total candles saved: ${totalSaved}`);
    return totalSaved;
}

// Clear day buffers (keep recent candles)
function clearDayBuffers() {
    for (const key of Object.keys(memoryBuffer)) {
        memoryBuffer[key].dayCandles = [];
    }
    console.log('Day buffers cleared (recent candles preserved)');
}

// ============================================================
// Hourly Checkpoint - Save current day's data every hour
// ============================================================
function saveHourlyCheckpoint() {
    const dayKey = getCurrentDayKey();
    console.log(`\n[${new Date().toISOString()}] Hourly checkpoint - saving to disk...`);

    let totalSaved = 0;

    for (const [category, assets] of Object.entries(ASSETS)) {
        for (const assetName of Object.keys(assets)) {
            const key = `${category}/${assetName}`;
            const buffer = memoryBuffer[key];

            if (buffer && buffer.dayCandles.length > 0) {
                saveDayToDisk(category, assetName, dayKey, buffer.dayCandles);
                totalSaved += buffer.dayCandles.length;
            }
        }
    }

    console.log(`[Hourly Checkpoint] Saved ${totalSaved} candles to disk`);
}

// ============================================================
// Check if it's time to save (midnight UTC)
// ============================================================
let lastSaveDay = getCurrentDayKey();

function checkEndOfDay() {
    const currentDay = getCurrentDayKey();

    if (currentDay !== lastSaveDay) {
        console.log(`\n*** Day changed from ${lastSaveDay} to ${currentDay} ***`);

        // Save yesterday's data
        saveAllToDisk();

        // Clear day buffers
        clearDayBuffers();

        lastSaveDay = currentDay;
    }
}

// ============================================================
// Main Loop
// ============================================================
async function main() {
    const { execSync } = require('child_process');

    console.log('============================================================');
    console.log('End-of-Day Data Collector - TradingView');
    console.log('============================================================');
    console.log(`Data directory: ${DATA_DIR}`);
    console.log(`Poll interval: ${POLL_INTERVAL / 1000} seconds`);
    console.log(`Recent candles buffer: ${RECENT_CANDLES_COUNT} candles`);
    console.log('============================================================');

    let totalAssets = 0;
    for (const [category, assets] of Object.entries(ASSETS)) {
        const assetList = Object.keys(assets).join(', ');
        console.log(`  ${category}: ${assetList}`);
        totalAssets += Object.keys(assets).length;
    }
    console.log(`\nTotal: ${totalAssets} assets`);
    console.log('============================================================');
    console.log('Data saved: hourly checkpoints + end of day (midnight UTC)');
    console.log('Recent candles are always available in memory for predictions');
    console.log('Press Ctrl+C to stop (will save current data first).\n');

    ensureDir(DATA_DIR);
    initializeBuffer();

    // Auto-backfill on startup to recover any gaps
    console.log('Running startup backfill to recover gaps...');
    try {
        execSync('node backfill_gaps.js', { cwd: __dirname, stdio: 'inherit' });
    } catch (e) {
        console.log('Backfill completed (or no gaps to fill)');
    }
    console.log('');

    let totalPolls = 0;
    let totalNewCandles = 0;

    // Initial poll
    const initialResult = await pollAllAssets();
    totalPolls++;
    totalNewCandles += initialResult.newCandles;

    // Poll every POLL_INTERVAL
    const pollInterval = setInterval(async () => {
        const result = await pollAllAssets();
        totalPolls++;
        totalNewCandles += result.newCandles;

        // Check if we need to save (new day)
        checkEndOfDay();
    }, POLL_INTERVAL);

    // Also check for end of day every minute (in case poll fails)
    const eodCheckInterval = setInterval(() => {
        checkEndOfDay();
    }, 60000);

    // Hourly checkpoint - save current day's data every hour
    const hourlyCheckpointInterval = setInterval(() => {
        saveHourlyCheckpoint();
    }, 60 * 60 * 1000); // Every hour

    // Graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n\nShutting down...');
        clearInterval(pollInterval);
        clearInterval(eodCheckInterval);
        clearInterval(hourlyCheckpointInterval);

        // Save current data before exit
        console.log('Saving current data before exit...');
        saveAllToDisk();

        console.log(`\nTotal polls: ${totalPolls}`);
        console.log(`Total new candles: ${totalNewCandles}`);
        process.exit(0);
    });
}

// Only run main if this is the entry point
if (require.main === module) {
    main().catch(err => {
        console.error('Error:', err.message);
        process.exit(1);
    });
}
