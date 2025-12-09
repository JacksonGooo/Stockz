'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge } from '@/components/ui';
import { StockList } from '@/components/stocks';

// Commodity symbols (TradingView format)
const COMMODITY_SYMBOLS = [
  'TVC:GOLD',      // Gold
  'TVC:SILVER',    // Silver
  'TVC:USOIL',     // Crude Oil
  'TVC:NATGAS',    // Natural Gas
  'COMEX:HG1!',    // Copper
  'TVC:PLATINUM',  // Platinum
];

interface CommodityAsset {
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

export default function CommoditiesPage() {
  const [commodities, setCommodities] = useState<CommodityAsset[]>([]);
  const [topGainers, setTopGainers] = useState<CommodityAsset[]>([]);
  const [topLosers, setTopLosers] = useState<CommodityAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchCommodityData() {
      try {
        setIsLoading(true);

        // Fetch commodity quotes from TradingView
        const response = await fetch(`/api/stocks/quotes?symbols=${COMMODITY_SYMBOLS.join(',')}`);
        const data = await response.json();

        if (data.quotes) {
          const commodityData: CommodityAsset[] = data.quotes.map((q: any) => {
            const symbol = extractSymbol(q.symbol);

            return {
              symbol,
              name: getDisplayName(symbol),
              sector: 'Commodities',
              currentPrice: q.price || 0,
              previousClose: q.previousClose || q.price,
              change: q.change || 0,
              changePercent: q.changePercent || 0,
              volume: q.volume || 0,
              marketCap: 0,
              high52Week: q.price * 1.3,
              low52Week: q.price * 0.7,
            };
          });

          setCommodities(commodityData);

          // Calculate top gainers (highest positive change)
          const gainers = commodityData.filter(c => c.changePercent > 0).sort((a, b) => b.changePercent - a.changePercent);
          setTopGainers(gainers.slice(0, 3));

          // Calculate top losers (most negative change first)
          const losers = commodityData.filter(c => c.changePercent < 0).sort((a, b) => a.changePercent - b.changePercent);
          setTopLosers(losers.slice(0, 3));
        }
      } catch (error) {
        console.error('Failed to fetch commodity data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchCommodityData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchCommodityData, 30000);
    return () => clearInterval(interval);
  }, []);

  const avgChangePercent = commodities.length > 0
    ? commodities.reduce((sum, c) => sum + c.changePercent, 0) / commodities.length
    : 0;

  const totalVolume = commodities.reduce((sum, c) => sum + c.volume * c.currentPrice, 0);

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Commodities Market
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Real-time precious metals, energy, and industrial commodities
        </p>
      </div>

      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard
          title="Total Assets"
          value={commodities.length.toString()}
          subtitle="Tracked commodities"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
          }
          color="yellow"
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
          value={formatLargeNumber(totalVolume)}
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

      {/* All commodities */}
      <StockList
        stocks={commodities}
        isLoading={isLoading}
        showDetails
        title="All Commodities"
      />
    </Layout>
  );
}

// Extract readable symbol from TradingView format
function extractSymbol(tvSymbol: string): string {
  const mapping: Record<string, string> = {
    'GOLD': 'GOLD',
    'SILVER': 'SILVER',
    'USOIL': 'OIL',
    'NATGAS': 'NATGAS',
    'HG1!': 'COPPER',
    'PLATINUM': 'PLATINUM',
  };

  for (const [key, value] of Object.entries(mapping)) {
    if (tvSymbol.includes(key)) return value;
  }

  return tvSymbol.split(':').pop() || tvSymbol;
}

// Helper function to get display name
function getDisplayName(symbol: string): string {
  const names: Record<string, string> = {
    'GOLD': 'Gold',
    'SILVER': 'Silver',
    'OIL': 'Crude Oil WTI',
    'NATGAS': 'Natural Gas',
    'COPPER': 'Copper',
    'PLATINUM': 'Platinum',
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
