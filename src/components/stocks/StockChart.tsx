'use client';

import { useState, useMemo } from 'react';
import { Button } from '../ui/Button';
import { StockHistoricalData } from '@/ai/types';

type TimeRange = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';

interface StockChartProps {
  data: StockHistoricalData[];
  symbol: string;
}

export function StockChart({ data, symbol }: StockChartProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('1W');

  const filteredData = useMemo(() => {
    const now = new Date();
    const ranges: Record<TimeRange, number> = {
      '1D': 1,
      '1W': 7,
      '1M': 30,
      '3M': 90,
      '6M': 180,
      '1Y': 365,
    };
    const daysBack = ranges[timeRange];
    const cutoff = new Date(now.getTime() - daysBack * 24 * 60 * 60 * 1000);

    return data.filter((d) => new Date(d.timestamp) >= cutoff);
  }, [data, timeRange]);

  const chartMetrics = useMemo(() => {
    if (filteredData.length === 0) return null;

    const prices = filteredData.map((d) => d.close);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const first = prices[0];
    const last = prices[prices.length - 1];
    const change = last - first;
    const changePercent = (change / first) * 100;

    return { min, max, range, first, last, change, changePercent };
  }, [filteredData]);

  const isPositive = chartMetrics ? chartMetrics.change >= 0 : true;

  const pathData = useMemo(() => {
    if (!chartMetrics || filteredData.length < 2) return '';

    const width = 100;
    const height = 100;
    const padding = 5;

    const points = filteredData.map((d, i) => {
      const x = padding + ((width - padding * 2) * i) / (filteredData.length - 1);
      const y =
        height -
        padding -
        ((d.close - chartMetrics.min) / chartMetrics.range) *
          (height - padding * 2);
      return `${x},${y}`;
    });

    return `M ${points.join(' L ')}`;
  }, [filteredData, chartMetrics]);

  const areaPath = useMemo(() => {
    if (!pathData) return '';
    return `${pathData} L 95,100 L 5,100 Z`;
  }, [pathData]);

  return (
    <div className="w-full">
      {/* Chart header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            {symbol} Price Chart
          </h3>
          {chartMetrics && (
            <p
              className={`text-sm font-medium ${
                isPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
              }`}
            >
              {isPositive ? '+' : ''}
              {chartMetrics.changePercent.toFixed(2)}% ({timeRange})
            </p>
          )}
        </div>

        {/* Time range selector */}
        <div className="flex gap-1 p-1 bg-zinc-100 dark:bg-zinc-800 rounded-lg">
          {(['1D', '1W', '1M', '3M', '6M', '1Y'] as TimeRange[]).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`
                px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200
                ${
                  timeRange === range
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-sm'
                    : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100'
                }
              `}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Chart area */}
      <div className="relative w-full h-64 bg-gradient-to-b from-zinc-50 to-white dark:from-zinc-900 dark:to-zinc-950 rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
        {filteredData.length > 1 ? (
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="w-full h-full"
          >
            {/* Gradient definition */}
            <defs>
              <linearGradient
                id={`gradient-${symbol}`}
                x1="0%"
                y1="0%"
                x2="0%"
                y2="100%"
              >
                <stop
                  offset="0%"
                  stopColor={isPositive ? '#10b981' : '#ef4444'}
                  stopOpacity="0.3"
                />
                <stop
                  offset="100%"
                  stopColor={isPositive ? '#10b981' : '#ef4444'}
                  stopOpacity="0"
                />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {[20, 40, 60, 80].map((y) => (
              <line
                key={y}
                x1="0"
                y1={y}
                x2="100"
                y2={y}
                stroke="currentColor"
                strokeOpacity="0.1"
                strokeWidth="0.2"
              />
            ))}

            {/* Area fill */}
            <path
              d={areaPath}
              fill={`url(#gradient-${symbol})`}
              className="transition-all duration-500"
            />

            {/* Line */}
            <path
              d={pathData}
              fill="none"
              stroke={isPositive ? '#10b981' : '#ef4444'}
              strokeWidth="0.8"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="transition-all duration-500"
            />
          </svg>
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-400">
            No data available for this time range
          </div>
        )}

        {/* Price labels */}
        {chartMetrics && (
          <>
            <div className="absolute top-2 left-3 text-xs font-mono text-zinc-500 dark:text-zinc-400">
              ${chartMetrics.max.toFixed(2)}
            </div>
            <div className="absolute bottom-2 left-3 text-xs font-mono text-zinc-500 dark:text-zinc-400">
              ${chartMetrics.min.toFixed(2)}
            </div>
          </>
        )}
      </div>

      {/* Volume bars */}
      <div className="mt-2 h-12 flex items-end gap-px rounded-lg overflow-hidden">
        {filteredData.slice(-50).map((d, i) => {
          const maxVol = Math.max(...filteredData.slice(-50).map((x) => x.volume));
          const height = (d.volume / maxVol) * 100;
          const dayPositive = d.close >= d.open;

          return (
            <div
              key={i}
              className={`flex-1 rounded-t transition-all duration-200 ${
                dayPositive
                  ? 'bg-emerald-300 dark:bg-emerald-700'
                  : 'bg-red-300 dark:bg-red-700'
              }`}
              style={{ height: `${height}%` }}
            />
          );
        })}
      </div>
      <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">Volume</p>
    </div>
  );
}
