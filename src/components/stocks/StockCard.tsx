'use client';

import { memo, useMemo } from 'react';
import Link from 'next/link';
import { Card } from '../ui/Card';
import { PriceChangeBadge } from '../ui/Badge';
import { Stock } from '@/ai/types';

interface StockCardProps {
  stock: Stock;
  showDetails?: boolean;
  isUpdating?: boolean;
}

// Memoized StockCard to prevent unnecessary re-renders
export const StockCard = memo(function StockCard({
  stock,
  showDetails = false,
  isUpdating = false,
}: StockCardProps) {
  const isPositive = stock.change >= 0;

  // Memoize chart data to avoid recalculating on every render
  const chartData = useMemo(
    () => generateMiniChart(stock.symbol, isPositive),
    [stock.symbol, isPositive]
  );

  return (
    <Link href={`/stock/${stock.symbol}`}>
      <Card
        hover
        variant="default"
        className={`
          group overflow-hidden transition-all duration-200
          ${isUpdating ? 'ring-2 ring-blue-400 dark:ring-blue-500' : ''}
        `}
      >
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div
              className={`
                w-10 h-10 flex-shrink-0 rounded-xl flex items-center justify-center
                font-bold text-sm transition-transform duration-300 group-hover:scale-110
                ${
                  isPositive
                    ? 'bg-gradient-to-br from-emerald-100 to-emerald-50 text-emerald-600 dark:from-emerald-900/30 dark:to-emerald-900/10 dark:text-emerald-400'
                    : 'bg-gradient-to-br from-red-100 to-red-50 text-red-600 dark:from-red-900/30 dark:to-red-900/10 dark:text-red-400'
                }
              `}
            >
              {stock.symbol.slice(0, 2)}
            </div>
            <div className="min-w-0">
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                {stock.symbol}
              </h3>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 truncate">
                {stock.name}
              </p>
            </div>
          </div>
          <PriceChangeBadge
            change={stock.change}
            changePercent={stock.changePercent}
            size="sm"
          />
        </div>

        <div className="flex items-end justify-between gap-2">
          <div className="min-w-0">
            <p className="text-lg font-bold text-zinc-900 dark:text-zinc-100 font-mono">
              ${stock.currentPrice.toFixed(2)}
            </p>
            {showDetails && (
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                Vol: {formatVolume(stock.volume)}
              </p>
            )}
          </div>

          {/* Mini sparkline */}
          <div className="w-16 h-8 flex items-end gap-0.5 flex-shrink-0">
            {chartData.map((height, i) => (
              <div
                key={i}
                className={`
                  w-1 rounded-t transition-all duration-300
                  ${
                    isPositive
                      ? 'bg-emerald-400 dark:bg-emerald-500'
                      : 'bg-red-400 dark:bg-red-500'
                  }
                `}
                style={{ height: `${height}%` }}
              />
            ))}
          </div>
        </div>

        {showDetails && (
          <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800 grid grid-cols-2 gap-2">
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Market Cap</p>
              <p className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
                {formatMarketCap(stock.marketCap)}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">52W Range</p>
              <p className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
                ${stock.low52Week.toFixed(0)} - ${stock.high52Week.toFixed(0)}
              </p>
            </div>
          </div>
        )}
      </Card>
    </Link>
  );
});

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

// Seeded random number generator for consistent values between server/client
function seededRandom(seed: string): () => number {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    const char = seed.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }

  return () => {
    hash = Math.sin(hash) * 10000;
    return hash - Math.floor(hash);
  };
}

function generateMiniChart(symbol: string, isPositive: boolean): number[] {
  const random = seededRandom(symbol);
  const points = 10;
  const base = isPositive ? 40 : 60;
  const trend = isPositive ? 3 : -3;

  return Array.from({ length: points }, (_, i) => {
    const noise = (random() - 0.5) * 20;
    return Math.max(10, Math.min(100, base + trend * i + noise));
  });
}
