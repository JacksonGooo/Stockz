/**
 * Stock Data Service
 * Provides stock data using the ML data provider
 */

import {
  Stock,
  StockHistoricalData,
  WatchlistItem,
} from './types';
import {
  fetchHistoricalData,
  fetchStockQuote,
  fetchMultipleQuotes,
  getSupportedSymbols,
  getStockInfoSync,
  HistoricalDataPoint,
} from './ml';

// Cache for all stocks to avoid redundant API calls
let stocksCache: { data: Stock[]; timestamp: number } | null = null;
const STOCKS_CACHE_DURATION = 30 * 1000; // 30 seconds

class StockService {
  private watchlist: WatchlistItem[] = [];

  /**
   * Get all available stocks - with caching
   */
  async getAllStocks(): Promise<Stock[]> {
    // Return cached data if fresh
    if (stocksCache && Date.now() - stocksCache.timestamp < STOCKS_CACHE_DURATION) {
      return stocksCache.data;
    }

    const symbols = getSupportedSymbols();
    const quotes = await fetchMultipleQuotes(symbols);

    const stocks = quotes.map((quote) => {
      const info = getStockInfoSync(quote.symbol);
      return {
        symbol: quote.symbol,
        name: info?.name || quote.name,
        sector: info?.sector || 'Unknown',
        currentPrice: quote.price,
        previousClose: quote.price - quote.change,
        change: quote.change,
        changePercent: quote.changePercent,
        volume: quote.volume,
        marketCap: quote.marketCap,
        high52Week: quote.high52Week,
        low52Week: quote.low52Week,
      };
    });

    // Cache the result
    stocksCache = { data: stocks, timestamp: Date.now() };
    return stocks;
  }

  /**
   * Get stock by symbol
   */
  async getStock(symbol: string): Promise<Stock | null> {
    const quote = await fetchStockQuote(symbol);
    if (!quote) return null;

    const info = getStockInfoSync(symbol);
    return {
      symbol: quote.symbol,
      name: info?.name || quote.name,
      sector: info?.sector || 'Unknown',
      currentPrice: quote.price,
      previousClose: quote.price - quote.change,
      change: quote.change,
      changePercent: quote.changePercent,
      volume: quote.volume,
      marketCap: quote.marketCap,
      high52Week: quote.high52Week,
      low52Week: quote.low52Week,
    };
  }

  /**
   * Search stocks by name or symbol
   */
  async searchStocks(query: string): Promise<Stock[]> {
    const allStocks = await this.getAllStocks();
    const lowerQuery = query.toLowerCase();

    return allStocks.filter(
      (s) =>
        s.symbol.toLowerCase().includes(lowerQuery) ||
        s.name.toLowerCase().includes(lowerQuery)
    );
  }

  /**
   * Get historical data for a stock
   */
  async getHistoricalData(
    symbol: string,
    days: number = 30
  ): Promise<StockHistoricalData[]> {
    const data = await fetchHistoricalData(symbol, days);

    return data.map((d: HistoricalDataPoint) => ({
      symbol,
      timestamp: d.timestamp,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }));
  }

  /**
   * Get stocks by sector
   */
  async getStocksBySector(sector: string): Promise<Stock[]> {
    const allStocks = await this.getAllStocks();
    return allStocks.filter((s) => s.sector === sector);
  }

  /**
   * Get top gainers
   */
  async getTopGainers(limit: number = 5): Promise<Stock[]> {
    const allStocks = await this.getAllStocks();
    return [...allStocks]
      .sort((a, b) => b.changePercent - a.changePercent)
      .slice(0, limit);
  }

  /**
   * Get top losers
   */
  async getTopLosers(limit: number = 5): Promise<Stock[]> {
    const allStocks = await this.getAllStocks();
    return [...allStocks]
      .sort((a, b) => a.changePercent - b.changePercent)
      .slice(0, limit);
  }

  /**
   * Get watchlist
   */
  getWatchlist(): WatchlistItem[] {
    return this.watchlist;
  }

  /**
   * Add to watchlist
   */
  async addToWatchlist(symbol: string): Promise<boolean> {
    const stock = await this.getStock(symbol);
    if (!stock) return false;

    if (!this.watchlist.find((w) => w.stock.symbol === symbol)) {
      this.watchlist.push({
        stock,
        addedAt: new Date(),
      });
    }
    return true;
  }

  /**
   * Remove from watchlist
   */
  removeFromWatchlist(symbol: string): boolean {
    const index = this.watchlist.findIndex((w) => w.stock.symbol === symbol);
    if (index > -1) {
      this.watchlist.splice(index, 1);
      return true;
    }
    return false;
  }
}

export const stockService = new StockService();
