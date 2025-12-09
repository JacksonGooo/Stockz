'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge } from '@/components/ui';
import { StockList } from '@/components/stocks';

// Forex currency pair symbols (TradingView format)
const CURRENCY_SYMBOLS = [
  'FX:EURUSD',    // Euro/US Dollar
  'FX:GBPUSD',    // British Pound/US Dollar
  'FX:USDJPY',    // US Dollar/Japanese Yen
  'FX:AUDUSD',    // Australian Dollar/US Dollar
  'FX:USDCAD',    // US Dollar/Canadian Dollar
  'FX:USDCHF',    // US Dollar/Swiss Franc
  'FX:NZDUSD',    // New Zealand Dollar/US Dollar
  'FX:EURGBP',    // Euro/British Pound
];

interface CurrencyAsset {
  symbol: string;
  name: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high52Week: number;
  low52Week: number;
  sector: string;
}

export default function CurrenciesPage() {
  const [currencies, setCurrencies] = useState<CurrencyAsset[]>([]);
  const [topGainers, setTopGainers] = useState<CurrencyAsset[]>([]);
  const [topLosers, setTopLosers] = useState<CurrencyAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchCurrencyData() {
      try {
        setIsLoading(true);

        // Fetch currency quotes from TradingView
        const response = await fetch(`/api/stocks/quotes?symbols=${CURRENCY_SYMBOLS.join(',')}`);
        const data = await response.json();

        if (data.quotes) {
          const currencyData: CurrencyAsset[] = data.quotes.map((q: any) => {
            const symbol = extractSymbol(q.symbol);

            return {
              symbol,
              name: getDisplayName(symbol),
              sector: 'Forex',
              currentPrice: q.price || 0,
              previousClose: q.previousClose || q.price,
              change: q.change || 0,
              changePercent: q.changePercent || 0,
              volume: q.volume || 0,
              marketCap: 0,
              high52Week: q.price * 1.1,
              low52Week: q.price * 0.9,
            };
          });

          setCurrencies(currencyData);

          // Calculate top gainers (highest positive change)
          const gainers = currencyData.filter(c => c.changePercent > 0).sort((a, b) => b.changePercent - a.changePercent);
          setTopGainers(gainers.slice(0, 3));

          // Calculate top losers (most negative change first)
          const losers = currencyData.filter(c => c.changePercent < 0).sort((a, b) => a.changePercent - b.changePercent);
          setTopLosers(losers.slice(0, 3));
        }
      } catch (error) {
        console.error('Failed to fetch currency data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchCurrencyData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchCurrencyData, 30000);
    return () => clearInterval(interval);
  }, []);

  const avgChangePercent = currencies.length > 0
    ? currencies.reduce((sum, c) => sum + c.changePercent, 0) / currencies.length
    : 0;

  const totalVolume = currencies.reduce((sum, c) => sum + c.volume, 0);

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Foreign Exchange Market
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Real-time forex currency pairs and exchange rates
        </p>
      </div>

      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard
          title="Currency Pairs"
          value={currencies.length.toString()}
          subtitle="Major forex pairs"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
            </svg>
          }
          color="blue"
        />
        <StatCard
          title="Market Trend"
          value={avgChangePercent >= 0 ? 'USD WEAK' : 'USD STRONG'}
          subtitle={`Avg: ${avgChangePercent.toFixed(4)}%`}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
          color={avgChangePercent >= 0 ? 'green' : 'red'}
        />
        <StatCard
          title="Market Status"
          value={isForexOpen() ? 'OPEN' : 'CLOSED'}
          subtitle={isForexOpen() ? '24/5 Trading' : 'Opens Sunday 5pm ET'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          color={isForexOpen() ? 'emerald' : 'yellow'}
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

      {/* All currencies */}
      <StockList
        stocks={currencies}
        isLoading={isLoading}
        showDetails
        title="All Currency Pairs"
      />
    </Layout>
  );
}

// Check if forex market is open
function isForexOpen(): boolean {
  const now = new Date();
  const day = now.getUTCDay();
  const hour = now.getUTCHours();

  // Forex is open Sun 10pm UTC to Fri 10pm UTC
  if (day === 0) return hour >= 22; // Sunday opens at 10pm UTC
  if (day === 6) return false; // Saturday closed
  if (day === 5) return hour < 22; // Friday closes at 10pm UTC
  return true; // Mon-Thu 24h
}

// Extract readable symbol from TradingView format
function extractSymbol(tvSymbol: string): string {
  return tvSymbol.split(':').pop() || tvSymbol;
}

// Helper function to get display name
function getDisplayName(symbol: string): string {
  const names: Record<string, string> = {
    'EURUSD': 'Euro / US Dollar',
    'GBPUSD': 'British Pound / US Dollar',
    'USDJPY': 'US Dollar / Japanese Yen',
    'AUDUSD': 'Australian Dollar / US Dollar',
    'USDCAD': 'US Dollar / Canadian Dollar',
    'USDCHF': 'US Dollar / Swiss Franc',
    'NZDUSD': 'New Zealand Dollar / US Dollar',
    'EURGBP': 'Euro / British Pound',
  };
  return names[symbol] || symbol;
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
