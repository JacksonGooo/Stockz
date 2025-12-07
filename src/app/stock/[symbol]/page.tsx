'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Layout } from '@/components/layout';
import { Card, CardHeader, CardTitle, Button, Badge, Select } from '@/components/ui';
import { StockChart, PredictionCard } from '@/components/stocks';
import { PredictionCandlestickChart } from '@/components/charts';
import { ChartSkeleton } from '@/components/ui/Skeleton';
import { useFocusedFetch } from '@/hooks';
import { formatTimeCST } from '@/lib/time';
import {
  stockService,
  predictionService,
  Stock,
  StockHistoricalData,
  PredictionResult,
  PredictionTimeframe,
} from '@/ai';

// Prediction candle data interface
interface PredictionCandleData {
  symbol: string;
  assetType: 'stock' | 'crypto';
  generatedAt: number;
  expiresAt: number;
  candles: Array<{
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    type: 'historical' | 'predicted';
    direction: 'up' | 'down';
  }>;
  metadata: {
    historicalCount: number;
    predictedCount: number;
    confidence: number;
    trend: string;
    ageMinutes: number;
    expiresInMinutes: number;
  };
}

interface LiveQuote {
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

// Fetch function for live stock data
async function fetchStockQuote(symbol: string): Promise<LiveQuote | null> {
  try {
    const response = await fetch(`/api/stocks/quote/${symbol}`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    console.error('Failed to fetch stock quote:', error);
    return null;
  }
}

export default function StockDetailPage() {
  const params = useParams();
  const symbol = params.symbol as string;

  const [stock, setStock] = useState<Stock | null>(null);
  const [historicalData, setHistoricalData] = useState<StockHistoricalData[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<PredictionTimeframe>('1w');
  const [isLoading, setIsLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isInWatchlist, setIsInWatchlist] = useState(false);
  const [priceFlash, setPriceFlash] = useState<'up' | 'down' | null>(null);
  const [predictionCandles, setPredictionCandles] = useState<PredictionCandleData | null>(null);
  const [isPredictionCandlesLoading, setIsPredictionCandlesLoading] = useState(false);

  // Handle price change flash effect
  const handlePriceChange = useCallback((direction: 'up' | 'down') => {
    setPriceFlash(direction);
    setTimeout(() => setPriceFlash(null), 500);
  }, []);

  // Fetch prediction candles
  const fetchPredictionCandles = useCallback(async (forceRegenerate = false) => {
    if (!symbol) return;
    setIsPredictionCandlesLoading(true);
    try {
      const url = `/api/predictions/${symbol}/candles?type=stock${forceRegenerate ? '&force=true' : ''}`;
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        setPredictionCandles(data);
      }
    } catch (error) {
      console.error('Failed to fetch prediction candles:', error);
    } finally {
      setIsPredictionCandlesLoading(false);
    }
  }, [symbol]);

  // Handle prediction candles refresh
  const handlePredictionRefresh = useCallback(async () => {
    if (!symbol) return;
    setIsPredictionCandlesLoading(true);
    try {
      // First regenerate
      await fetch(`/api/predictions/${symbol}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'stock' }),
      });
      // Then fetch new data
      await fetchPredictionCandles(false);
    } catch (error) {
      console.error('Failed to refresh predictions:', error);
      setIsPredictionCandlesLoading(false);
    }
  }, [symbol, fetchPredictionCandles]);

  // Live quote fetching with 2 second interval
  const {
    data: liveQuote,
    isLive,
    setIsLive,
    lastUpdated,
    isLoading: isLiveLoading,
  } = useFocusedFetch<LiveQuote>({
    symbol,
    fetchFn: fetchStockQuote,
    intervalMs: 2000,
    enabled: !!symbol && !isLoading,
    onPriceChange: handlePriceChange,
    getPrice: (quote) => quote.price,
  });

  useEffect(() => {
    // Load stock data first (fast) - don't wait for ML prediction
    async function fetchStockData() {
      setIsLoading(true);
      try {
        const [stockData, history] = await Promise.all([
          stockService.getStock(symbol),
          stockService.getHistoricalData(symbol, 365),
        ]);

        setStock(stockData);
        setHistoricalData(history);

        // Check watchlist
        const watchlist = stockService.getWatchlist();
        setIsInWatchlist(watchlist.some((w) => w.stock.symbol === symbol));
      } catch (error) {
        console.error('Failed to fetch stock data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    // Load prediction separately (slow - involves ML model)
    async function fetchPrediction() {
      try {
        const pred = await predictionService.getPrediction(symbol, selectedTimeframe);
        setPrediction(pred);
      } catch (error) {
        console.error('Failed to fetch prediction:', error);
      }
    }

    if (symbol) {
      fetchStockData();
      fetchPrediction();
    }
  }, [symbol, selectedTimeframe]);

  // Fetch prediction candles on mount and when symbol changes
  useEffect(() => {
    if (symbol) {
      fetchPredictionCandles();
    }
  }, [symbol, fetchPredictionCandles]);

  // Use live quote data if available, otherwise use initial stock data
  const displayPrice = liveQuote?.price ?? stock?.currentPrice ?? 0;
  const displayChange = liveQuote?.change ?? stock?.change ?? 0;
  const displayChangePercent = liveQuote?.changePercent ?? stock?.changePercent ?? 0;
  const displayVolume = liveQuote?.volume ?? stock?.volume ?? 0;
  const displayOpen = liveQuote?.open ?? stock?.previousClose ?? 0;
  const displayHigh = liveQuote?.high ?? 0;
  const displayLow = liveQuote?.low ?? 0;

  const handleGetPrediction = async () => {
    setIsPredicting(true);
    try {
      const pred = await predictionService.getPrediction(symbol, selectedTimeframe);
      setPrediction(pred);
    } catch (error) {
      console.error('Failed to get prediction:', error);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleWatchlistToggle = async () => {
    if (isInWatchlist) {
      stockService.removeFromWatchlist(symbol);
      setIsInWatchlist(false);
    } else {
      await stockService.addToWatchlist(symbol);
      setIsInWatchlist(true);
    }
  };

  if (isLoading) {
    return (
      <Layout>
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
          <div className="h-4 w-64 bg-zinc-200 dark:bg-zinc-800 rounded mb-8" />
          <ChartSkeleton />
        </div>
      </Layout>
    );
  }

  if (!stock) {
    return (
      <Layout>
        <div className="text-center py-20">
          <svg
            className="w-16 h-16 mx-auto text-zinc-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
            Stock Not Found
          </h2>
          <p className="text-zinc-500 dark:text-zinc-400 mb-4">
            We couldn&apos;t find a stock with symbol &quot;{symbol}&quot;
          </p>
          <Link href="/">
            <Button variant="primary">Back to Dashboard</Button>
          </Link>
        </div>
      </Layout>
    );
  }

  const isPositive = displayChange >= 0;

  return (
    <Layout>
      {/* Breadcrumb */}
      <nav className="mb-6">
        <ol className="flex items-center gap-2 text-sm">
          <li>
            <Link
              href="/"
              className="text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
            >
              Dashboard
            </Link>
          </li>
          <li className="text-zinc-400">/</li>
          <li className="text-zinc-900 dark:text-zinc-100 font-medium">
            {stock.symbol}
          </li>
        </ol>
      </nav>

      {/* Stock header */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
        <div className="flex items-center gap-4">
          <div
            className={`
              w-16 h-16 rounded-2xl flex items-center justify-center
              font-bold text-2xl
              ${
                isPositive
                  ? 'bg-gradient-to-br from-emerald-100 to-emerald-50 text-emerald-600 dark:from-emerald-900/30 dark:to-emerald-900/10 dark:text-emerald-400'
                  : 'bg-gradient-to-br from-red-100 to-red-50 text-red-600 dark:from-red-900/30 dark:to-red-900/10 dark:text-red-400'
              }
            `}
          >
            {stock.symbol.slice(0, 2)}
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                {stock.symbol}
              </h1>
              <Badge variant="default" size="sm">
                {stock.sector}
              </Badge>
            </div>
            <p className="text-zinc-500 dark:text-zinc-400">{stock.name}</p>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-3">
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
            <Button
              variant={isInWatchlist ? 'secondary' : 'ghost'}
              onClick={handleWatchlistToggle}
              leftIcon={
                <svg
                  className={`w-5 h-5 ${isInWatchlist ? 'fill-current' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                  />
                </svg>
              }
            >
              {isInWatchlist ? 'In Watchlist' : 'Add to Watchlist'}
            </Button>
          </div>
          {/* Last updated */}
          {lastUpdated && (
            <span className="text-xs text-zinc-400">
              Last update: {formatTimeCST(lastUpdated)}
            </span>
          )}
        </div>
      </div>

      {/* Price section */}
      <Card variant="elevated" className="mb-8">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">
              Current Price
            </p>
            <div className="flex items-baseline gap-3">
              <span
                className={`
                  text-4xl font-bold font-mono text-zinc-900 dark:text-zinc-100
                  transition-all duration-300
                  ${priceFlash === 'up' ? 'text-emerald-500 scale-105' : ''}
                  ${priceFlash === 'down' ? 'text-red-500 scale-105' : ''}
                `}
              >
                ${displayPrice.toFixed(2)}
              </span>
              <span
                className={`text-lg font-semibold ${
                  isPositive
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {isPositive ? '+' : ''}
                {displayChange.toFixed(2)} ({isPositive ? '+' : ''}
                {displayChangePercent.toFixed(2)}%)
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-8">
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Open</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                ${displayOpen.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Volume</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                {formatVolume(displayVolume)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Market Cap</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                {formatMarketCap(stock.marketCap)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">52W Range</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                ${stock.low52Week.toFixed(0)} - ${stock.high52Week.toFixed(0)}
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* 90-Minute Prediction Chart */}
      <div className="mb-8">
        {isPredictionCandlesLoading && !predictionCandles ? (
          <Card className="p-6">
            <div className="animate-pulse">
              <div className="h-6 w-48 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
              <div className="h-[300px] bg-zinc-200 dark:bg-zinc-800 rounded" />
            </div>
          </Card>
        ) : predictionCandles ? (
          <PredictionCandlestickChart
            symbol={stock.symbol}
            candles={predictionCandles.candles}
            generatedAt={predictionCandles.generatedAt}
            confidence={predictionCandles.metadata.confidence}
            trend={predictionCandles.metadata.trend}
            ageMinutes={predictionCandles.metadata.ageMinutes}
            expiresInMinutes={predictionCandles.metadata.expiresInMinutes}
            onRefresh={handlePredictionRefresh}
            isLoading={isPredictionCandlesLoading}
          />
        ) : (
          <Card className="p-6">
            <div className="flex items-center justify-center h-64">
              <p className="text-zinc-500">Loading prediction chart...</p>
            </div>
          </Card>
        )}
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Historical Chart */}
        <div className="lg:col-span-2">
          <Card>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4 px-4 pt-4">
              Historical Performance
            </h3>
            <StockChart data={historicalData} symbol={stock.symbol} />
          </Card>
        </div>

        {/* Prediction panel */}
        <div className="space-y-6">
          {/* Prediction controls */}
          <Card>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
              Get AI Prediction
            </h3>
            <div className="space-y-4">
              <Select
                label="Timeframe"
                value={selectedTimeframe}
                onChange={(e) =>
                  setSelectedTimeframe(e.target.value as PredictionTimeframe)
                }
                options={[
                  { value: '1d', label: '1 Day' },
                  { value: '1w', label: '1 Week' },
                  { value: '1m', label: '1 Month' },
                  { value: '3m', label: '3 Months' },
                  { value: '6m', label: '6 Months' },
                  { value: '1y', label: '1 Year' },
                ]}
              />
              <Button
                variant="primary"
                className="w-full"
                onClick={handleGetPrediction}
                isLoading={isPredicting}
                leftIcon={
                  !isPredicting && (
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
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                      />
                    </svg>
                  )
                }
              >
                Generate Prediction
              </Button>
            </div>
          </Card>

          {/* Prediction result */}
          {prediction && <PredictionCard prediction={prediction} />}
        </div>
      </div>
    </Layout>
  );
}

function formatVolume(volume: number): string {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toString();
}

function formatMarketCap(marketCap: number): string {
  if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`;
  if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`;
  if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`;
  return `$${marketCap}`;
}
