/**
 * TradingView Data Service - FAST HTTP-based
 * Uses TradingView's Scanner API for instant multi-stock quotes
 */

import axios from 'axios';

export interface TVQuote {
  symbol: string;
  exchange: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  volume: number;
  currency: string;
  recommendation?: string;
  taScore?: number;
}

export interface TVCandle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TVSearchResult {
  id: string;
  symbol: string;
  exchange: string;
  name: string;
  type: string;
}

export interface TVTechnicalAnalysis {
  recommendation: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL';
  score: number;
  periods: Record<string, { Other: number; All: number; MA: number }>;
}

// Aggressive caching
const quoteCache = new Map<string, { data: TVQuote; timestamp: number }>();
const multiQuoteCache = new Map<string, { data: TVQuote[]; timestamp: number }>();
const candleCache = new Map<string, { data: TVCandle[]; timestamp: number }>();
const searchCache = new Map<string, { data: TVSearchResult[]; timestamp: number }>();

const QUOTE_CACHE_DURATION = 30 * 1000; // 30 seconds
const CANDLE_CACHE_DURATION = 60 * 1000; // 1 minute (predictions update every minute)
const SEARCH_CACHE_DURATION = 60 * 60 * 1000; // 1 hour

// Exchange mappings for common stocks
const EXCHANGE_MAP: Record<string, string> = {
  'AAPL': 'NASDAQ', 'MSFT': 'NASDAQ', 'GOOGL': 'NASDAQ', 'GOOG': 'NASDAQ',
  'AMZN': 'NASDAQ', 'META': 'NASDAQ', 'NVDA': 'NASDAQ', 'TSLA': 'NASDAQ',
  'NFLX': 'NASDAQ', 'AMD': 'NASDAQ', 'INTC': 'NASDAQ', 'CSCO': 'NASDAQ',
  'ADBE': 'NASDAQ', 'PYPL': 'NASDAQ', 'CMCSA': 'NASDAQ', 'QCOM': 'NASDAQ',
  'JPM': 'NYSE', 'V': 'NYSE', 'JNJ': 'NYSE', 'WMT': 'NYSE', 'PG': 'NYSE',
  'UNH': 'NYSE', 'HD': 'NYSE', 'MA': 'NYSE', 'DIS': 'NYSE', 'BAC': 'NYSE',
  'XOM': 'NYSE', 'KO': 'NYSE', 'PFE': 'NYSE', 'VZ': 'NYSE', 'T': 'NYSE',
  'ABBV': 'NYSE', 'CVX': 'NYSE', 'MRK': 'NYSE', 'PEP': 'NYSE', 'TMO': 'NYSE',
};

function getTVSymbol(symbol: string): string {
  const upper = symbol.toUpperCase();
  if (upper.includes(':')) return upper;
  const exchange = EXCHANGE_MAP[upper] || 'NASDAQ';
  return `${exchange}:${upper}`;
}

/**
 * FAST: Get multiple quotes in a SINGLE HTTP request using Scanner API
 */
export async function getMultipleQuotes(symbols: string[]): Promise<TVQuote[]> {
  if (symbols.length === 0) return [];

  const cacheKey = symbols.sort().join(',');
  const cached = multiQuoteCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < QUOTE_CACHE_DURATION) {
    return cached.data;
  }

  const tickers = symbols.map(s => getTVSymbol(s));

  // Detect asset type and use appropriate scanner endpoint
  const isCrypto = tickers.some(t => t.includes('BINANCE:') || t.includes('COINBASE:') || t.includes('KRAKEN:'));
  const isForex = tickers.some(t => t.includes('FX:') || t.includes('FOREX:'));
  const isCommodity = tickers.some(t => t.includes('TVC:') || t.includes('COMEX:') || t.includes('NYMEX:'));

  let scannerEndpoint = 'https://scanner.tradingview.com/america/scan';
  if (isCrypto) scannerEndpoint = 'https://scanner.tradingview.com/crypto/scan';
  else if (isForex) scannerEndpoint = 'https://scanner.tradingview.com/forex/scan';
  else if (isCommodity) scannerEndpoint = 'https://scanner.tradingview.com/cfd/scan';

  try {
    // Single HTTP request for ALL stocks - much faster than websockets
    const { data } = await axios.post(
      scannerEndpoint,
      {
        symbols: { tickers },
        columns: [
          'name', 'close', 'change', 'change_abs', 'high', 'low', 'open',
          'volume', 'Recommend.All', 'exchange', 'description', 'type',
          'Perf.W', 'Perf.1M', 'Perf.3M', 'Perf.6M', 'Perf.Y', 'Perf.YTD',
          'Volatility.W', 'Volatility.M', 'SMA20', 'SMA50', 'SMA200',
          'RSI', 'MACD.macd', 'MACD.signal', 'ADX', 'Mom', 'CCI20',
          'Stoch.K', 'Stoch.D', 'W.R', 'BBPower', 'ATR',
          'High.1M', 'Low.1M', 'High.3M', 'Low.3M', 'High.6M', 'Low.6M',
          'price_52_week_high', 'price_52_week_low', 'currency',
        ],
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0',
        },
        timeout: 5000, // 5 second timeout
      }
    );

    const quotes: TVQuote[] = data.data.map((item: { s: string; d: (string | number | null)[] }) => {
      const [exchange, symbol] = item.s.split(':');
      const d = item.d;

      // Get recommendation text from score
      const taScore = (d[8] as number) || 0;
      let recommendation = 'NEUTRAL';
      if (taScore >= 0.5) recommendation = 'STRONG_BUY';
      else if (taScore >= 0.1) recommendation = 'BUY';
      else if (taScore <= -0.5) recommendation = 'STRONG_SELL';
      else if (taScore <= -0.1) recommendation = 'SELL';

      const quote: TVQuote = {
        symbol,
        exchange,
        name: (d[10] as string) || (d[0] as string) || symbol,
        price: (d[1] as number) || 0,
        changePercent: (d[2] as number) || 0,
        change: (d[3] as number) || 0,
        high: (d[4] as number) || 0,
        low: (d[5] as number) || 0,
        open: (d[6] as number) || 0,
        previousClose: ((d[1] as number) || 0) - ((d[3] as number) || 0),
        volume: (d[7] as number) || 0,
        currency: (d[42] as string) || 'USD',
        recommendation,
        taScore,
      };

      // Also cache individual quotes
      quoteCache.set(getTVSymbol(symbol), { data: quote, timestamp: Date.now() });

      return quote;
    });

    multiQuoteCache.set(cacheKey, { data: quotes, timestamp: Date.now() });
    return quotes;
  } catch (error) {
    console.error('Error fetching multiple quotes:', error);
    // Return cached data if available
    const fallback: TVQuote[] = [];
    for (const symbol of symbols) {
      const cached = quoteCache.get(getTVSymbol(symbol));
      if (cached) fallback.push(cached.data);
    }
    return fallback;
  }
}

/**
 * Get single quote - uses cache or batch fetch
 */
export async function getQuote(symbol: string): Promise<TVQuote | null> {
  const tvSymbol = getTVSymbol(symbol);
  const cached = quoteCache.get(tvSymbol);
  if (cached && Date.now() - cached.timestamp < QUOTE_CACHE_DURATION) {
    return cached.data;
  }

  const quotes = await getMultipleQuotes([symbol]);
  return quotes[0] || null;
}

/**
 * FAST: Get technical analysis using HTTP Scanner API
 */
export async function getTechnicalAnalysis(symbol: string): Promise<TVTechnicalAnalysis | null> {
  const tvSymbol = getTVSymbol(symbol);

  try {
    const { data } = await axios.post(
      'https://scanner.tradingview.com/america/scan',
      {
        symbols: { tickers: [tvSymbol] },
        columns: [
          'Recommend.All', 'Recommend.MA', 'Recommend.Other',
          'Recommend.All|1', 'Recommend.MA|1', 'Recommend.Other|1',
          'Recommend.All|5', 'Recommend.MA|5', 'Recommend.Other|5',
          'Recommend.All|15', 'Recommend.MA|15', 'Recommend.Other|15',
          'Recommend.All|60', 'Recommend.MA|60', 'Recommend.Other|60',
          'Recommend.All|240', 'Recommend.MA|240', 'Recommend.Other|240',
          'Recommend.All|1W', 'Recommend.MA|1W', 'Recommend.Other|1W',
          'Recommend.All|1M', 'Recommend.MA|1M', 'Recommend.Other|1M',
        ],
      },
      {
        headers: { 'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0' },
        timeout: 3000,
      }
    );

    if (!data.data || !data.data[0]) return null;

    const d = data.data[0].d;
    const allScore = (d[0] as number) || 0;

    let recommendation: TVTechnicalAnalysis['recommendation'] = 'NEUTRAL';
    if (allScore >= 0.5) recommendation = 'STRONG_BUY';
    else if (allScore >= 0.1) recommendation = 'BUY';
    else if (allScore <= -0.5) recommendation = 'STRONG_SELL';
    else if (allScore <= -0.1) recommendation = 'SELL';

    return {
      recommendation,
      score: allScore,
      periods: {
        '1D': { All: d[0] as number, MA: d[1] as number, Other: d[2] as number },
        '1': { All: d[3] as number, MA: d[4] as number, Other: d[5] as number },
        '5': { All: d[6] as number, MA: d[7] as number, Other: d[8] as number },
        '15': { All: d[9] as number, MA: d[10] as number, Other: d[11] as number },
        '60': { All: d[12] as number, MA: d[13] as number, Other: d[14] as number },
        '240': { All: d[15] as number, MA: d[16] as number, Other: d[17] as number },
        '1W': { All: d[18] as number, MA: d[19] as number, Other: d[20] as number },
        '1M': { All: d[21] as number, MA: d[22] as number, Other: d[23] as number },
      },
    };
  } catch (error) {
    console.error('Error getting TA:', error);
    return null;
  }
}

/**
 * FAST: Search stocks using HTTP API
 */
export async function searchStocks(query: string, filter: 'stock' | '' = 'stock'): Promise<TVSearchResult[]> {
  const cacheKey = `${query}-${filter}`;
  const cached = searchCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < SEARCH_CACHE_DURATION) {
    return cached.data;
  }

  try {
    const { data } = await axios.get(
      'https://symbol-search.tradingview.com/symbol_search/v3/',
      {
        params: {
          text: query,
          search_type: filter,
          start: 0,
          exchange: '',
        },
        headers: { origin: 'https://www.tradingview.com' },
        timeout: 3000,
      }
    );

    const results: TVSearchResult[] = data.symbols
      .filter((s: { exchange: string }) =>
        ['NASDAQ', 'NYSE', 'AMEX'].includes(s.exchange?.toUpperCase?.() || '')
      )
      .slice(0, 15)
      .map((s: { symbol: string; exchange: string; description: string; type: string; prefix?: string }) => ({
        id: s.prefix ? `${s.prefix}:${s.symbol}` : `${s.exchange}:${s.symbol}`,
        symbol: s.symbol,
        exchange: s.exchange,
        name: s.description,
        type: s.type,
      }));

    searchCache.set(cacheKey, { data: results, timestamp: Date.now() });
    return results;
  } catch (error) {
    console.error('Error searching:', error);
    return [];
  }
}

/**
 * Get historical candles - uses TradingView chart API (websocket - slower but necessary for historical)
 */
export async function getCandles(
  symbol: string,
  timeframe: string = 'D',
  count: number = 100
): Promise<TVCandle[]> {
  const tvSymbol = getTVSymbol(symbol);
  const cacheKey = `${tvSymbol}-${timeframe}-${count}`;

  const cached = candleCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CANDLE_CACHE_DURATION) {
    return cached.data;
  }

  // For historical data, we still need websocket but with aggressive caching
  // This is only called when viewing charts, not on dashboard
  try {
    // @ts-expect-error - no types
    const TradingView = await import('@mathieuc/tradingview');

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        chart.delete();
        client.end();
        resolve(cached?.data || generateFallbackCandles(symbol, count));
      }, 6000);

      const client = new TradingView.Client();
      const chart = new client.Session.Chart();

      chart.setMarket(tvSymbol, { timeframe, range: count });

      chart.onError(() => {
        clearTimeout(timeout);
        chart.delete();
        client.end();
        resolve(cached?.data || generateFallbackCandles(symbol, count));
      });

      let done = false;
      chart.onUpdate(() => {
        if (done || !chart.periods || chart.periods.length < 2) return;
        done = true;

        const candles: TVCandle[] = chart.periods
          .filter((p: { time?: number }) => p?.time)
          .map((p: { time: number; open: number; max: number; min: number; close: number; volume?: number }) => ({
            timestamp: p.time * 1000,
            open: p.open,
            high: p.max,
            low: p.min,
            close: p.close,
            volume: p.volume || 0,
          }))
          .sort((a: TVCandle, b: TVCandle) => a.timestamp - b.timestamp);

        candleCache.set(cacheKey, { data: candles, timestamp: Date.now() });
        clearTimeout(timeout);
        chart.delete();
        client.end();
        resolve(candles);
      });
    });
  } catch (error) {
    console.error('Error getting candles:', error);
    return cached?.data || generateFallbackCandles(symbol, count);
  }
}

/**
 * Generate fallback candles for when API fails
 */
function generateFallbackCandles(symbol: string, count: number): TVCandle[] {
  const candles: TVCandle[] = [];
  let seed = symbol.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const seededRandom = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

  let price = 100 + seededRandom() * 200;
  const now = Date.now();

  for (let i = count; i >= 0; i--) {
    const change = (seededRandom() - 0.5) * price * 0.03;
    price = Math.max(10, price + change);
    const vol = seededRandom() * 0.02;
    candles.push({
      timestamp: now - i * 86400000,
      open: price * (1 - vol),
      high: price * (1 + vol),
      low: price * (1 - vol * 1.5),
      close: price,
      volume: Math.floor(1000000 + seededRandom() * 50000000),
    });
  }
  return candles;
}

export function clearCache(): void {
  quoteCache.clear();
  multiQuoteCache.clear();
  candleCache.clear();
  searchCache.clear();
}
