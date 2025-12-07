'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Layout } from '@/components/layout';
import { Card, Button, Badge } from '@/components/ui';
import { PredictionCandlestickChart } from '@/components/charts';
import { formatTimeCST } from '@/lib/time';

interface CryptoData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  recommendation?: string;
  taScore?: number;
}

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

const REFRESH_INTERVAL = 2000; // 2 seconds - focused fetching for detail page

export default function CryptoDetailPage() {
  const params = useParams();
  const symbol = (params.symbol as string).toUpperCase();

  const [crypto, setCrypto] = useState<CryptoData | null>(null);
  const [prevPrice, setPrevPrice] = useState<number | null>(null);
  const [priceFlash, setPriceFlash] = useState<'up' | 'down' | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isInWatchlist, setIsInWatchlist] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isLive, setIsLive] = useState(true);
  const [predictionCandles, setPredictionCandles] = useState<PredictionCandleData | null>(null);
  const [isPredictionCandlesLoading, setIsPredictionCandlesLoading] = useState(false);

  // Get base symbol without USDT suffix
  const getBaseSymbol = (sym: string) => {
    return sym.replace('USDT', '').replace('USD', '');
  };

  const fetchCryptoData = useCallback(async (showLoading = false) => {
    if (showLoading) setIsLoading(true);
    try {
      const response = await fetch(`/api/crypto?symbols=${symbol}`);
      const data = await response.json();

      if (data.quotes && data.quotes.length > 0) {
        const newCrypto = data.quotes[0];

        // Flash price change indicator
        if (crypto && newCrypto.price !== crypto.price) {
          setPrevPrice(crypto.price);
          setPriceFlash(newCrypto.price > crypto.price ? 'up' : 'down');
          setTimeout(() => setPriceFlash(null), 1000);
        }

        setCrypto(newCrypto);
        setLastUpdated(new Date());
      }
    } catch (error) {
      console.error('Failed to fetch crypto data:', error);
    } finally {
      if (showLoading) setIsLoading(false);
    }
  }, [symbol, crypto]);

  // Fetch prediction candles
  const fetchPredictionCandles = useCallback(async (forceRegenerate = false) => {
    if (!symbol) return;
    setIsPredictionCandlesLoading(true);
    try {
      const url = `/api/predictions/${symbol}/candles?type=crypto${forceRegenerate ? '&force=true' : ''}`;
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
        body: JSON.stringify({ type: 'crypto' }),
      });
      // Then fetch new data
      await fetchPredictionCandles(false);
    } catch (error) {
      console.error('Failed to refresh predictions:', error);
      setIsPredictionCandlesLoading(false);
    }
  }, [symbol, fetchPredictionCandles]);

  // Initial fetch
  useEffect(() => {
    fetchCryptoData(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // Live updates
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      fetchCryptoData(false);
    }, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, [isLive, fetchCryptoData]);

  // Fetch prediction candles on mount and when symbol changes
  useEffect(() => {
    if (symbol) {
      fetchPredictionCandles();
    }
  }, [symbol, fetchPredictionCandles]);

  const handleWatchlistToggle = () => {
    setIsInWatchlist(!isInWatchlist);
  };

  if (isLoading) {
    return (
      <Layout>
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
          <div className="h-4 w-64 bg-zinc-200 dark:bg-zinc-800 rounded mb-8" />
          <Card className="h-64" />
        </div>
      </Layout>
    );
  }

  if (!crypto) {
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
            Cryptocurrency Not Found
          </h2>
          <p className="text-zinc-500 dark:text-zinc-400 mb-4">
            We couldn&apos;t find data for &quot;{symbol}&quot;
          </p>
          <Link href="/crypto">
            <Button variant="primary">Back to Crypto</Button>
          </Link>
        </div>
      </Layout>
    );
  }

  const isPositive = crypto.change >= 0;
  const baseSymbol = getBaseSymbol(crypto.symbol);

  const getCryptoColor = (sym: string) => {
    const base = getBaseSymbol(sym);
    const colors: Record<string, string> = {
      BTC: 'from-orange-500/20 to-amber-500/20 text-orange-600 dark:text-orange-400',
      ETH: 'from-indigo-500/20 to-purple-500/20 text-indigo-600 dark:text-indigo-400',
      BNB: 'from-yellow-500/20 to-amber-500/20 text-yellow-600 dark:text-yellow-400',
      XRP: 'from-slate-500/20 to-gray-500/20 text-slate-600 dark:text-slate-400',
      SOL: 'from-purple-500/20 to-pink-500/20 text-purple-600 dark:text-purple-400',
      ADA: 'from-blue-500/20 to-cyan-500/20 text-blue-600 dark:text-blue-400',
      DOGE: 'from-amber-500/20 to-yellow-500/20 text-amber-600 dark:text-amber-400',
      AVAX: 'from-red-500/20 to-rose-500/20 text-red-600 dark:text-red-400',
      DOT: 'from-pink-500/20 to-rose-500/20 text-pink-600 dark:text-pink-400',
      LINK: 'from-blue-500/20 to-indigo-500/20 text-blue-600 dark:text-blue-400',
    };
    return colors[base] || 'from-zinc-500/20 to-gray-500/20 text-zinc-600 dark:text-zinc-400';
  };

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
          <li>
            <Link
              href="/crypto"
              className="text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
            >
              Crypto
            </Link>
          </li>
          <li className="text-zinc-400">/</li>
          <li className="text-zinc-900 dark:text-zinc-100 font-medium">
            {baseSymbol}
          </li>
        </ol>
      </nav>

      {/* Crypto header */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
        <div className="flex items-center gap-4">
          <div
            className={`
              w-16 h-16 rounded-2xl flex items-center justify-center
              font-bold text-2xl bg-gradient-to-br ${getCryptoColor(crypto.symbol)}
            `}
          >
            {baseSymbol.charAt(0)}
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                {baseSymbol}
              </h1>
              <Badge variant="info" size="sm">
                Crypto
              </Badge>
            </div>
            <p className="text-zinc-500 dark:text-zinc-400">{crypto.name}</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
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
      </div>

      {/* Price section */}
      <Card variant="elevated" className="mb-8">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Current Price
              </p>
              {/* Live indicator */}
              <div className="flex items-center gap-2">
                {isLive && (
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                  </span>
                )}
                <button
                  onClick={() => setIsLive(!isLive)}
                  className={`text-xs px-2 py-0.5 rounded transition-all ${
                    isLive
                      ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                      : 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'
                  }`}
                >
                  {isLive ? 'Live' : 'Paused'}
                </button>
                {lastUpdated && (
                  <span className="text-xs text-zinc-400">
                    {formatTimeCST(lastUpdated)}
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-baseline gap-3">
              <span
                className={`text-4xl font-bold font-mono text-zinc-900 dark:text-zinc-100 transition-all duration-300 ${
                  priceFlash === 'up'
                    ? 'text-emerald-500 dark:text-emerald-400'
                    : priceFlash === 'down'
                      ? 'text-red-500 dark:text-red-400'
                      : ''
                }`}
              >
                ${formatPrice(crypto.price)}
              </span>
              <span
                className={`text-lg font-semibold ${
                  isPositive
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {isPositive ? '+' : ''}
                {crypto.changePercent.toFixed(2)}%
              </span>
              {priceFlash && (
                <span className={`text-sm ${priceFlash === 'up' ? 'text-emerald-500' : 'text-red-500'}`}>
                  {priceFlash === 'up' ? '▲' : '▼'}
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-8">
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">24h High</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                ${formatPrice(crypto.high)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">24h Low</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                ${formatPrice(crypto.low)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">24h Volume</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                ${formatVolume(crypto.volume)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Open</p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                ${formatPrice(crypto.open)}
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
            symbol={getBaseSymbol(crypto.symbol)}
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
        {/* Stats */}
        <div className="lg:col-span-2">
          <Card>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
              Market Statistics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-900">
                <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Price Change (24h)</p>
                <p className={`text-xl font-bold ${isPositive ? 'text-emerald-600' : 'text-red-600'}`}>
                  {isPositive ? '+' : ''}${formatPrice(Math.abs(crypto.change))}
                </p>
              </div>
              <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-900">
                <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Change %</p>
                <p className={`text-xl font-bold ${isPositive ? 'text-emerald-600' : 'text-red-600'}`}>
                  {isPositive ? '+' : ''}{crypto.changePercent.toFixed(2)}%
                </p>
              </div>
              <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-900">
                <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Previous Close</p>
                <p className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
                  ${formatPrice(crypto.previousClose)}
                </p>
              </div>
              <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-900">
                <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">24h Range</p>
                <p className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
                  ${formatPrice(crypto.low)} - ${formatPrice(crypto.high)}
                </p>
              </div>
            </div>
          </Card>
        </div>

        {/* Analysis panel */}
        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
              Technical Analysis
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-zinc-600 dark:text-zinc-400">Signal</span>
                <Badge
                  variant={
                    crypto.recommendation?.includes('BUY')
                      ? 'success'
                      : crypto.recommendation?.includes('SELL')
                        ? 'danger'
                        : 'default'
                  }
                >
                  {crypto.recommendation || 'NEUTRAL'}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-600 dark:text-zinc-400">TA Score</span>
                <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100">
                  {((crypto.taScore || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Technical analysis based on multiple indicators including RSI, MACD, and moving averages.
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
              Quick Info
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-zinc-600 dark:text-zinc-400">Exchange</span>
                <span className="font-medium text-zinc-900 dark:text-zinc-100">Binance</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-600 dark:text-zinc-400">Pair</span>
                <span className="font-medium text-zinc-900 dark:text-zinc-100">{baseSymbol}/USDT</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-600 dark:text-zinc-400">Type</span>
                <span className="font-medium text-zinc-900 dark:text-zinc-100">Cryptocurrency</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </Layout>
  );
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (price >= 1) return price.toFixed(2);
  if (price >= 0.01) return price.toFixed(4);
  return price.toFixed(6);
}

function formatVolume(volume: number): string {
  if (volume >= 1e12) return `${(volume / 1e12).toFixed(2)}T`;
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toFixed(2);
}
