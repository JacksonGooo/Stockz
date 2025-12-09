'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge } from '@/components/ui';
import { StockList } from '@/components/stocks';
import { HistoricalChartViewer } from '@/components/charts';

// Major stock symbols
const STOCK_SYMBOLS = [
  'AAPL',  // Apple
  'MSFT',  // Microsoft
  'GOOGL', // Google
  'AMZN',  // Amazon
  'NVDA',  // NVIDIA
  'TSLA',  // Tesla
  'META',  // Meta
  'JPM',   // JPMorgan
  'V',     // Visa
  'WMT',   // Walmart
  'UNH',   // UnitedHealth
  'MA',    // Mastercard
];

interface StockAsset {
  symbol: string;
  name: string;
  currentPrice: number;
  previousClose: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high52Week: number;
  low52Week: number;
  sector: string;
}

export default function StocksPage() {
  const [stocks, setStocks] = useState<StockAsset[]>([]);
  const [topGainers, setTopGainers] = useState<StockAsset[]>([]);
  const [topLosers, setTopLosers] = useState<StockAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchStockData() {
      try {
        setIsLoading(true);

        // Fetch stock quotes
        const response = await fetch(`/api/stocks/quotes?symbols=${STOCK_SYMBOLS.join(',')}`);
        const data = await response.json();

        if (data.quotes) {
          const stockData: StockAsset[] = data.quotes.map((q: any) => ({
            symbol: q.symbol,
            name: q.name,
            sector: getSector(q.symbol),
            currentPrice: q.price || 0,
            previousClose: q.previousClose || q.price,
            change: q.change || 0,
            changePercent: q.changePercent || 0,
            volume: q.volume || 0,
            marketCap: 0,
            high52Week: q.price * 1.3,
            low52Week: q.price * 0.7,
          }));

          setStocks(stockData);

          // Calculate top gainers (highest positive change)
          const gainers = stockData.filter(s => s.changePercent > 0).sort((a, b) => b.changePercent - a.changePercent);
          setTopGainers(gainers.slice(0, 3));

          // Calculate top losers (most negative change first)
          const losers = stockData.filter(s => s.changePercent < 0).sort((a, b) => a.changePercent - b.changePercent);
          setTopLosers(losers.slice(0, 3));
        }
      } catch (error) {
        console.error('Failed to fetch stock data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchStockData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchStockData, 30000);
    return () => clearInterval(interval);
  }, []);

  const totalMarketCap = stocks.reduce((sum, stock) => sum + stock.marketCap, 0);
  const avgChangePercent = stocks.length > 0
    ? stocks.reduce((sum, s) => sum + s.changePercent, 0) / stocks.length
    : 0;

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Stock Market
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Real-time stock prices and market data
        </p>
      </div>

      {/* Historical Chart Viewer */}
      <HistoricalChartViewer category="Stock Market" />

      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard
          title="Total Stocks"
          value={stocks.length.toString()}
          subtitle="Tracked stocks"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          }
          color="blue"
        />
        <StatCard
          title="Market Trend"
          value={avgChangePercent >= 0 ? 'BULLISH' : 'BEARISH'}
          subtitle={`Avg: ${avgChangePercent.toFixed(2)}%`}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
          color={avgChangePercent >= 0 ? 'green' : 'red'}
        />
        <StatCard
          title="24h Volume"
          value={formatLargeNumber(stocks.reduce((sum, s) => sum + s.volume * s.currentPrice, 0))}
          subtitle="Trading volume"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          color="purple"
        />
      </div>

      {/* Top movers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Top Gainers
            </h2>
            <Badge variant="success" size="sm">
              24h
            </Badge>
          </div>
          <StockList stocks={topGainers} isLoading={isLoading} compact />
        </div>
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Top Losers
            </h2>
            <Badge variant="danger" size="sm">
              24h
            </Badge>
          </div>
          <StockList stocks={topLosers} isLoading={isLoading} compact />
        </div>
      </div>

      {/* All stocks */}
      <StockList
        stocks={stocks}
        isLoading={isLoading}
        showDetails
        title="All Stocks"
      />
    </Layout>
  );
}

// Helper function to get sector
function getSector(symbol: string): string {
  const sectors: Record<string, string> = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'AMZN': 'Consumer Cyclical',
    'NVDA': 'Technology',
    'TSLA': 'Automotive',
    'META': 'Technology',
    'JPM': 'Financial',
    'V': 'Financial',
    'WMT': 'Consumer Defensive',
    'UNH': 'Healthcare',
    'MA': 'Financial',
  };
  return sectors[symbol] || 'Unknown';
}

// Format large numbers
function formatLargeNumber(num: number): string {
  if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
  return `$${num.toFixed(2)}`;
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
