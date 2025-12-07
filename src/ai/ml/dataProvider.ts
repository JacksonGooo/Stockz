/**
 * Stock Data Provider
 * Fetches real stock data from Finnhub API via Next.js API routes
 */

import { OHLCV } from './indicators';

export interface StockQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high52Week: number;
  low52Week: number;
  logo?: string;
  industry?: string;
}

export interface HistoricalDataPoint extends OHLCV {
  timestamp: Date;
  adjClose: number;
}

// Cache for API responses to reduce calls - longer durations for performance
const quoteCache = new Map<string, { data: StockQuote; timestamp: number }>();
const historicalCache = new Map<string, { data: HistoricalDataPoint[]; timestamp: number }>();
const CACHE_DURATION = 2 * 60 * 1000; // 2 minutes for quotes
const HISTORICAL_CACHE_DURATION = 15 * 60 * 1000; // 15 minutes for historical

// Default watchlist symbols - reduced for faster loading
const DEFAULT_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'
];

// Store dynamically added symbols
let dynamicSymbols: Set<string> = new Set(DEFAULT_SYMBOLS);

/**
 * Check if running on server or client
 */
function getBaseUrl(): string {
  if (typeof window !== 'undefined') {
    return ''; // Client-side: use relative URL
  }
  // Server-side: use absolute URL
  return process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';
}

/**
 * Fetch historical stock data from API
 */
export async function fetchHistoricalData(
  symbol: string,
  days: number = 365
): Promise<HistoricalDataPoint[]> {
  const cacheKey = `${symbol}-${days}`;
  const cached = historicalCache.get(cacheKey);

  if (cached && Date.now() - cached.timestamp < HISTORICAL_CACHE_DURATION) {
    return cached.data;
  }

  try {
    const baseUrl = getBaseUrl();
    const response = await fetch(
      `${baseUrl}/api/stocks/candles/${encodeURIComponent(symbol)}?days=${days}`
    );

    if (!response.ok) {
      console.warn(`Failed to fetch historical data for ${symbol}`);
      return generateFallbackHistory(symbol, days);
    }

    const json = await response.json();

    if (json.error || !json.candles) {
      return generateFallbackHistory(symbol, days);
    }

    const data: HistoricalDataPoint[] = json.candles.map((c: {
      timestamp: number;
      open: number;
      high: number;
      low: number;
      close: number;
      volume: number;
    }) => ({
      timestamp: new Date(c.timestamp),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume,
      adjClose: c.close,
    }));

    historicalCache.set(cacheKey, { data, timestamp: Date.now() });
    return data;
  } catch (error) {
    console.error(`Error fetching historical data for ${symbol}:`, error);
    return generateFallbackHistory(symbol, days);
  }
}

/**
 * Generate fallback historical data when API fails
 */
function generateFallbackHistory(symbol: string, days: number): HistoricalDataPoint[] {
  const data: HistoricalDataPoint[] = [];
  const basePrice = 100 + (symbol.charCodeAt(0) % 200);

  let seed = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const seededRandom = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  let currentPrice = basePrice;
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);

    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const change = (seededRandom() - 0.5) * 0.04 * currentPrice;
    currentPrice += change;
    currentPrice = Math.max(currentPrice, basePrice * 0.5);

    const volatility = 0.015 + seededRandom() * 0.01;
    const open = currentPrice * (1 + (seededRandom() - 0.5) * volatility);
    const high = Math.max(open, currentPrice) * (1 + seededRandom() * volatility);
    const low = Math.min(open, currentPrice) * (1 - seededRandom() * volatility);

    data.push({
      timestamp: new Date(date),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(currentPrice.toFixed(2)),
      volume: Math.floor(10000000 + seededRandom() * 50000000),
      adjClose: Number(currentPrice.toFixed(2)),
    });
  }

  return data;
}

/**
 * Get current stock quote from API (now using Alpha Vantage)
 */
export async function fetchStockQuote(symbol: string): Promise<StockQuote | null> {
  const upperSymbol = symbol.toUpperCase();
  const cached = quoteCache.get(upperSymbol);

  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.data;
  }

  try {
    const baseUrl = getBaseUrl();
    const response = await fetch(
      `${baseUrl}/api/stocks/quote/${encodeURIComponent(upperSymbol)}`
    );

    if (!response.ok) {
      console.warn(`Failed to fetch quote for ${upperSymbol}`);
      return null;
    }

    const json = await response.json();

    if (json.error) {
      return null;
    }

    // Alpha Vantage provides 52-week high/low directly from company overview
    const high52Week = json.high52Week || json.price * 1.2;
    const low52Week = json.low52Week || json.price * 0.8;

    const quote: StockQuote = {
      symbol: upperSymbol,
      name: json.name || upperSymbol,
      price: json.price,
      change: json.change || 0,
      changePercent: json.changePercent || 0,
      volume: json.volume || 0,
      marketCap: json.marketCap || 0,
      high52Week,
      low52Week,
      logo: json.logo,
      industry: json.industry,
    };

    quoteCache.set(upperSymbol, { data: quote, timestamp: Date.now() });

    // Add to dynamic symbols
    dynamicSymbols.add(upperSymbol);

    return quote;
  } catch (error) {
    console.error(`Error fetching quote for ${upperSymbol}:`, error);
    return null;
  }
}

/**
 * Search for stocks by query
 */
export async function searchStocks(query: string): Promise<Array<{
  symbol: string;
  name: string;
  type: string;
}>> {
  if (!query || query.length < 1) {
    return [];
  }

  try {
    const baseUrl = getBaseUrl();
    const response = await fetch(
      `${baseUrl}/api/stocks/search?q=${encodeURIComponent(query)}`
    );

    if (!response.ok) {
      return [];
    }

    const json = await response.json();
    return json.results || [];
  } catch (error) {
    console.error('Error searching stocks:', error);
    return [];
  }
}

/**
 * Get list of all tracked symbols (default + dynamically added)
 */
export function getSupportedSymbols(): string[] {
  return Array.from(dynamicSymbols);
}

/**
 * Get default symbols for initial dashboard load
 */
export function getDefaultSymbols(): string[] {
  return [...DEFAULT_SYMBOLS];
}

/**
 * Add a symbol to the tracked list
 */
export function addSymbol(symbol: string): void {
  dynamicSymbols.add(symbol.toUpperCase());
}

/**
 * Remove a symbol from the tracked list
 */
export function removeSymbol(symbol: string): void {
  const upper = symbol.toUpperCase();
  // Don't remove default symbols
  if (!DEFAULT_SYMBOLS.includes(upper)) {
    dynamicSymbols.delete(upper);
  }
}

/**
 * Get stock info by symbol (fetches from API if not cached)
 */
export async function getStockInfo(symbol: string): Promise<{ name: string; sector: string } | null> {
  const quote = await fetchStockQuote(symbol);
  if (!quote) return null;

  return {
    name: quote.name,
    sector: quote.industry || 'Unknown',
  };
}

/**
 * Get stock info synchronously from cache (for compatibility)
 */
export function getStockInfoSync(symbol: string): { name: string; sector: string } | null {
  const cached = quoteCache.get(symbol.toUpperCase());
  if (!cached) return null;

  return {
    name: cached.data.name,
    sector: cached.data.industry || 'Unknown',
  };
}

/**
 * Fetch multiple stock quotes at once - FAST batch API
 */
export async function fetchMultipleQuotes(symbols: string[]): Promise<StockQuote[]> {
  if (symbols.length === 0) return [];

  try {
    const baseUrl = getBaseUrl();
    const response = await fetch(
      `${baseUrl}/api/stocks/quotes?symbols=${symbols.join(',')}`
    );

    if (!response.ok) {
      console.warn('Failed to fetch batch quotes');
      return [];
    }

    const json = await response.json();

    if (!json.quotes) return [];

    // Map API response to StockQuote format and cache individually
    const results: StockQuote[] = json.quotes.map((q: {
      symbol: string;
      name: string;
      price: number;
      change: number;
      changePercent: number;
      volume: number;
      high: number;
      low: number;
    }) => {
      const quote: StockQuote = {
        symbol: q.symbol,
        name: q.name || q.symbol,
        price: q.price,
        change: q.change || 0,
        changePercent: q.changePercent || 0,
        volume: q.volume || 0,
        marketCap: 0,
        high52Week: q.price * 1.2,
        low52Week: q.price * 0.8,
      };

      // Cache individual quote
      quoteCache.set(q.symbol.toUpperCase(), { data: quote, timestamp: Date.now() });

      return quote;
    });

    return results;
  } catch (error) {
    console.error('Error fetching batch quotes:', error);
    return [];
  }
}

/**
 * Stream real-time price updates
 */
export function createPriceStream(
  symbol: string,
  callback: (price: number, change: number) => void,
  intervalMs: number = 10000
): () => void {
  let previousPrice: number | null = null;

  const fetchAndUpdate = async () => {
    const quote = await fetchStockQuote(symbol);
    if (quote) {
      const change = previousPrice !== null ? quote.price - previousPrice : quote.change;
      previousPrice = quote.price;
      callback(quote.price, change);
    }
  };

  // Initial fetch
  fetchAndUpdate();

  // Set up interval
  const interval = setInterval(fetchAndUpdate, intervalMs);

  return () => clearInterval(interval);
}

/**
 * Get market hours status
 */
export function isMarketOpen(): boolean {
  const now = new Date();
  const day = now.getDay();
  const hour = now.getUTCHours();
  const minute = now.getUTCMinutes();

  // Market is open Mon-Fri, 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
  if (day === 0 || day === 6) return false;

  const currentMinutes = hour * 60 + minute;
  const openMinutes = 14 * 60 + 30; // 14:30 UTC
  const closeMinutes = 21 * 60; // 21:00 UTC

  return currentMinutes >= openMinutes && currentMinutes < closeMinutes;
}

/**
 * Clear all caches
 */
export function clearCache(): void {
  quoteCache.clear();
  historicalCache.clear();
}
