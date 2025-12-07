'use client';

import { useState, useEffect, useMemo, useCallback, memo } from 'react';
import { Layout } from '@/components/layout';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@/components/ui';
import { StockList, PredictionCard } from '@/components/stocks';
import { useCyclingFetch } from '@/hooks';
import { formatTimeCST } from '@/lib/time';
import { stockService, predictionService, Stock, PredictionResult, MarketSentiment } from '@/ai';

// Default stock symbols
const DEFAULT_STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA'];

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

export default function Dashboard() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [topGainers, setTopGainers] = useState<Stock[]>([]);
  const [topLosers, setTopLosers] = useState<Stock[]>([]);
  const [featuredPrediction, setFeaturedPrediction] = useState<PredictionResult | null>(null);
  const [sentiment, setSentiment] = useState<MarketSentiment | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Live stock cycling (3 stocks every 5 seconds - reduced frequency for performance)
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
    intervalMs: 5000, // Increased from 3000 for better performance
    getSymbol: (stock) => stock.symbol,
  });

  // Memoize data transformations to avoid recalculating on every render
  const liveStocksAsStock = useMemo<Stock[]>(() =>
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
    })),
    [liveStocks]
  );

  // Memoize gainers and losers calculations - filter by actual gain/loss
  const liveGainers = useMemo(() =>
    [...liveStocksAsStock]
      .filter(s => s.changePercent > 0) // Only positive changes
      .sort((a, b) => b.changePercent - a.changePercent)
      .slice(0, 3),
    [liveStocksAsStock]
  );

  const liveLosers = useMemo(() =>
    [...liveStocksAsStock]
      .filter(s => s.changePercent < 0) // Only negative changes
      .sort((a, b) => a.changePercent - b.changePercent)
      .slice(0, 3),
    [liveStocksAsStock]
  );

  useEffect(() => {
    // Load stock data first (fast) - don't wait for ML predictions
    async function fetchStockData() {
      try {
        const [allStocks, gainers, losers] = await Promise.all([
          stockService.getAllStocks(),
          stockService.getTopGainers(3),
          stockService.getTopLosers(3),
        ]);

        setStocks(allStocks);
        setTopGainers(gainers);
        setTopLosers(losers);
      } catch (error) {
        console.error('Failed to fetch stock data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    // Load ML predictions separately (slow - involves model training)
    async function fetchPredictions() {
      try {
        const [pred, sent] = await Promise.all([
          predictionService.getPrediction('NVDA', '1w'),
          predictionService.getMarketSentiment(),
        ]);

        setFeaturedPrediction(pred);
        setSentiment(sent);
      } catch (error) {
        console.error('Failed to fetch predictions:', error);
      }
    }

    // Run both in parallel but don't block stocks on predictions
    fetchStockData();
    fetchPredictions();
  }, []);

  // Memoize display data selection
  const displayStocks = useMemo(() =>
    liveStocks.length > 0 ? liveStocksAsStock : stocks,
    [liveStocks.length, liveStocksAsStock, stocks]
  );

  const displayGainers = useMemo(() =>
    liveStocks.length > 0 ? liveGainers : topGainers,
    [liveStocks.length, liveGainers, topGainers]
  );

  const displayLosers = useMemo(() =>
    liveStocks.length > 0 ? liveLosers : topLosers,
    [liveStocks.length, liveLosers, topLosers]
  );

  // Toggle handler - uses isLive from closure
  const handleToggleLive = useCallback(() => {
    setIsLive(!isLive);
  }, [setIsLive, isLive]);

  return (
    <Layout>
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Dashboard
          </h1>
          <p className="text-zinc-500 dark:text-zinc-400">
            AI-powered stock predictions and market analysis
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
              onClick={handleToggleLive}
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

      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          title="Total Stocks"
          value={stocks.length.toString()}
          subtitle="Being tracked"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          color="blue"
        />
        <StatCard
          title="Market Sentiment"
          value={sentiment?.overall.toUpperCase() || '...'}
          subtitle={`Score: ${sentiment?.score || 0}`}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
          color={sentiment?.overall === 'bullish' ? 'green' : sentiment?.overall === 'bearish' ? 'red' : 'yellow'}
        />
        <StatCard
          title="AI Accuracy"
          value="76%"
          subtitle="Last 30 days"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          }
          color="purple"
        />
        <StatCard
          title="Predictions Made"
          value="1,247"
          subtitle="This month"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          color="emerald"
        />
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Featured prediction */}
        <div className="lg:col-span-2">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
            Featured Prediction
          </h2>
          {featuredPrediction ? (
            <PredictionCard prediction={featuredPrediction} />
          ) : (
            <Card className="h-64 flex items-center justify-center">
              <div className="animate-pulse text-zinc-400">Loading prediction...</div>
            </Card>
          )}
        </div>

        {/* Quick actions */}
        <div>
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
            Quick Actions
          </h2>
          <Card>
            <div className="space-y-3">
              <Button variant="primary" className="w-full justify-start">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                New Prediction
              </Button>
              <Button variant="secondary" className="w-full justify-start">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                </svg>
                Add to Watchlist
              </Button>
              <Button variant="ghost" className="w-full justify-start">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                View History
              </Button>
            </div>

            <div className="mt-6 pt-6 border-t border-zinc-200 dark:border-zinc-800">
              <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-3">
                Model Status
              </h3>
              <div className="flex items-center gap-2 mb-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span className="text-sm text-zinc-600 dark:text-zinc-400">Online</span>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-500">
                Last trained: 3 days ago
              </p>
              <p className="text-xs text-zinc-500 dark:text-zinc-500">
                Data points: 183,247
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* Top movers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Top Gainers
            </h2>
            <Badge variant="success" size="sm">
              Today
            </Badge>
          </div>
          <StockList
            stocks={displayGainers}
            isLoading={isLoading && isLiveLoading}
            highlightSymbols={currentSymbols}
          />
        </div>
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Top Losers
            </h2>
            <Badge variant="danger" size="sm">
              Today
            </Badge>
          </div>
          <StockList
            stocks={displayLosers}
            isLoading={isLoading && isLiveLoading}
            highlightSymbols={currentSymbols}
          />
        </div>
      </div>

      {/* All stocks */}
      <StockList
        stocks={displayStocks}
        isLoading={isLoading && isLiveLoading}
        showDetails
        title="All Stocks"
        highlightSymbols={currentSymbols}
      />
    </Layout>
  );
}

// Stat card component
interface StatCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'emerald';
}

function StatCard({ title, value, subtitle, icon, color }: StatCardProps) {
  const colorStyles = {
    blue: 'from-blue-500/10 to-blue-500/5 text-blue-600 dark:text-blue-400',
    green: 'from-emerald-500/10 to-emerald-500/5 text-emerald-600 dark:text-emerald-400',
    red: 'from-red-500/10 to-red-500/5 text-red-600 dark:text-red-400',
    yellow: 'from-amber-500/10 to-amber-500/5 text-amber-600 dark:text-amber-400',
    purple: 'from-violet-500/10 to-violet-500/5 text-violet-600 dark:text-violet-400',
    emerald: 'from-emerald-500/10 to-emerald-500/5 text-emerald-600 dark:text-emerald-400',
  };

  return (
    <Card variant="default" padding="md">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            {value}
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
            {subtitle}
          </p>
        </div>
        <div
          className={`p-3 rounded-xl bg-gradient-to-br ${colorStyles[color]}`}
        >
          {icon}
        </div>
      </div>
    </Card>
  );
}
