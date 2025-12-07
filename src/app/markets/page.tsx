'use client';

import { useState, useEffect, useMemo } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge, Input } from '@/components/ui';
import { StockList } from '@/components/stocks';
import { useCyclingFetch } from '@/hooks';
import { formatTimeCST } from '@/lib/time';
import { stockService, Stock } from '@/ai';

const SECTORS = [
  'All',
  'Technology',
  'Consumer Cyclical',
  'Financial Services',
  'Healthcare',
  'Energy',
];

// Default stock symbols for the markets page
const DEFAULT_STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.A', 'JPM', 'V'];

interface StockQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  volume: number;
  recommendation?: string;
}

// Fetch function for stocks
async function fetchStockQuotes(symbols: string[]): Promise<StockQuote[]> {
  try {
    const response = await fetch(`/api/stocks/quotes?symbols=${symbols.join(',')}`);
    const data = await response.json();
    return data.quotes || [];
  } catch (error) {
    console.error('Failed to fetch stock quotes:', error);
    return [];
  }
}

export default function MarketsPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSector, setSelectedSector] = useState('All');
  const [isLoading, setIsLoading] = useState(true);

  // Live stock cycling (3 stocks every 3 seconds)
  const {
    data: liveStocks,
    isLive,
    setIsLive,
    currentSymbols,
    lastUpdated,
    isLoading: isLiveLoading,
  } = useCyclingFetch<StockQuote>({
    symbols: DEFAULT_STOCK_SYMBOLS,
    fetchFn: fetchStockQuotes,
    cycleSize: 3,
    intervalMs: 3000,
    getSymbol: (stock) => stock.symbol,
  });

  // Convert live stocks to Stock format (memoized to prevent recalculation)
  const liveStocksAsStock: Stock[] = useMemo(() =>
    liveStocks.map((q) => ({
      symbol: q.symbol,
      name: q.name,
      currentPrice: q.price,
      change: q.change,
      changePercent: q.changePercent,
      volume: q.volume,
      previousClose: q.previousClose,
      sector: 'Technology',
      marketCap: 0,
      peRatio: 0,
      divYield: 0,
      high52Week: q.high,
      low52Week: q.low,
    })), [liveStocks]);

  useEffect(() => {
    async function fetchStocks() {
      try {
        const allStocks = await stockService.getAllStocks();
        setStocks(allStocks);
      } catch (error) {
        console.error('Failed to fetch stocks:', error);
      } finally {
        setIsLoading(false);
      }
    }
    fetchStocks();
  }, []);

  // Use live data when available
  const displayStocks = liveStocks.length > 0 ? liveStocksAsStock : stocks;

  // Filter stocks (memoized to prevent recalculation on every render)
  const filteredStocks = useMemo(() => {
    let result = displayStocks;

    // Filter by sector
    if (selectedSector !== 'All') {
      result = result.filter((s) => s.sector === selectedSector);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (s) =>
          s.symbol.toLowerCase().includes(query) ||
          s.name.toLowerCase().includes(query)
      );
    }

    return result;
  }, [displayStocks, selectedSector, searchQuery]);

  // Calculate market stats from display stocks (memoized)
  const marketStats = useMemo(() => ({
    totalMarketCap: displayStocks.reduce((sum, s) => sum + s.marketCap, 0),
    avgChange: displayStocks.length
      ? displayStocks.reduce((sum, s) => sum + s.changePercent, 0) / displayStocks.length
      : 0,
    gainers: displayStocks.filter((s) => s.change > 0).length,
    losers: displayStocks.filter((s) => s.change < 0).length,
  }), [displayStocks]);

  return (
    <Layout>
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Traditional Stocks
          </h1>
          <p className="text-zinc-500 dark:text-zinc-400">
            Browse and analyze stocks across different sectors
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-4">
            {/* Live indicator */}
            <div className="flex items-center gap-2">
              {isLive && (
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
              )}
              <span className="text-sm text-zinc-600 dark:text-zinc-400">
                {isLive ? 'Live' : 'Paused'}
              </span>
            </div>
            {/* Toggle button */}
            <button
              onClick={() => setIsLive(!isLive)}
              className={`
                px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                ${isLive
                  ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                  : 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'
                }
              `}
            >
              {isLive ? 'Pause' : 'Resume'}
            </button>
          </div>
          {/* Cycling indicator */}
          {isLive && currentSymbols.length > 0 && (
            <div className="text-xs text-zinc-400 flex items-center gap-1">
              <span className="animate-pulse">Updating:</span>
              <span className="font-mono text-blue-500 dark:text-blue-400">
                {currentSymbols.join(', ')}
              </span>
            </div>
          )}
          {/* Last updated */}
          {lastUpdated && (
            <span className="text-xs text-zinc-400">
              Last update: {formatTimeCST(lastUpdated)}
            </span>
          )}
        </div>
      </div>

      {/* Market overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Total Market Cap</p>
          <p className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
            ${(marketStats.totalMarketCap / 1e12).toFixed(2)}T
          </p>
        </Card>
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Avg Change</p>
          <p
            className={`text-xl font-bold ${
              marketStats.avgChange >= 0
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-red-600 dark:text-red-400'
            }`}
          >
            {marketStats.avgChange >= 0 ? '+' : ''}
            {marketStats.avgChange.toFixed(2)}%
          </p>
        </Card>
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Gainers</p>
          <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
            {marketStats.gainers}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Losers</p>
          <p className="text-xl font-bold text-red-600 dark:text-red-400">
            {marketStats.losers}
          </p>
        </Card>
      </div>

      {/* Filters */}
      <Card className="mb-8">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <Input
              placeholder="Search stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={
                <svg
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              }
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            {SECTORS.map((sector) => (
              <button
                key={sector}
                onClick={() => setSelectedSector(sector)}
                className={`
                  px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200
                  ${
                    selectedSector === sector
                      ? 'bg-blue-600 text-white'
                      : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                  }
                `}
              >
                {sector}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Results count */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Showing {filteredStocks.length} of {stocks.length} stocks
        </p>
        {selectedSector !== 'All' && (
          <Badge variant="info" size="sm">
            {selectedSector}
          </Badge>
        )}
      </div>

      {/* Stock list */}
      <StockList
        stocks={filteredStocks}
        isLoading={isLoading && isLiveLoading}
        showDetails
        highlightSymbols={currentSymbols}
      />
    </Layout>
  );
}
