'use client';

import { memo, useCallback, useMemo, useState } from 'react';
import { Card } from '@/components/ui';
import { formatChartTimeCST, formatDateTimeCST } from '@/lib/time';

interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  type: 'historical' | 'predicted';
  direction: 'up' | 'down';
}

interface PredictionCandlestickChartProps {
  symbol: string;
  candles: CandleData[];
  generatedAt: number;
  confidence: number;
  trend: string;
  ageMinutes: number;
  expiresInMinutes: number;
  onRefresh?: () => void;
  isLoading?: boolean;
}

// Color scheme
const COLORS = {
  historicalUp: '#10B981', // Green
  historicalDown: '#EF4444', // Red
  predictedUp: '#F97316', // Orange
  predictedDown: '#3B82F6', // Blue
  gridLine: '#374151',
  nowLine: '#8B5CF6', // Purple
  text: '#9CA3AF',
};

export const PredictionCandlestickChart = memo(function PredictionCandlestickChart({
  symbol,
  candles,
  generatedAt,
  confidence,
  trend,
  ageMinutes,
  expiresInMinutes,
  onRefresh,
  isLoading = false,
}: PredictionCandlestickChartProps) {
  const [hoveredCandle, setHoveredCandle] = useState<CandleData | null>(null);

  // Chart dimensions
  const width = 900;
  const height = 400;
  const padding = { top: 40, right: 60, bottom: 60, left: 70 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Calculate chart metrics
  const chartData = useMemo(() => {
    if (candles.length === 0) return null;

    const prices = candles.flatMap(c => [c.high, c.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const pricePadding = priceRange * 0.05;

    const historicalCandles = candles.filter(c => c.type === 'historical');
    const predictedCandles = candles.filter(c => c.type === 'predicted');

    // Find the NOW divider position
    const lastHistoricalIndex = historicalCandles.length - 1;

    return {
      minPrice: minPrice - pricePadding,
      maxPrice: maxPrice + pricePadding,
      priceRange: priceRange + pricePadding * 2,
      historicalCount: historicalCandles.length,
      predictedCount: predictedCandles.length,
      lastHistoricalIndex,
      totalCandles: candles.length,
    };
  }, [candles]);

  // Calculate candle positions and sizes
  const candleWidth = useMemo(() => {
    if (!chartData) return 8;
    const totalCandles = chartData.totalCandles;
    const availableWidth = chartWidth - 20; // Leave some margin
    return Math.max(4, Math.min(12, availableWidth / totalCandles - 2));
  }, [chartData, chartWidth]);

  // Scale functions
  const xScale = useCallback((index: number) => {
    if (!chartData || chartData.totalCandles === 0) return 0;
    const spacing = chartWidth / chartData.totalCandles;
    return padding.left + spacing * index + spacing / 2;
  }, [chartData, chartWidth, padding.left]);

  const yScale = useCallback((price: number) => {
    if (!chartData || chartData.priceRange === 0) return padding.top + chartHeight / 2;
    const normalized = (price - chartData.minPrice) / chartData.priceRange;
    return padding.top + chartHeight * (1 - normalized);
  }, [chartData, chartHeight, padding.top]);

  // Get candle color
  const getCandleColor = (candle: CandleData) => {
    if (candle.type === 'historical') {
      return candle.direction === 'up' ? COLORS.historicalUp : COLORS.historicalDown;
    } else {
      return candle.direction === 'up' ? COLORS.predictedUp : COLORS.predictedDown;
    }
  };

  // Format price
  const formatPrice = (price: number) => {
    if (price >= 1000) return price.toFixed(0);
    if (price >= 100) return price.toFixed(1);
    if (price >= 1) return price.toFixed(2);
    return price.toFixed(4);
  };

  // Format time in CST
  const formatTime = (timestamp: number) => {
    return formatChartTimeCST(timestamp);
  };

  // Generate price axis labels
  const priceLabels = useMemo(() => {
    if (!chartData) return [];
    const labels = [];
    const step = chartData.priceRange / 5;
    for (let i = 0; i <= 5; i++) {
      const price = chartData.minPrice + step * i;
      labels.push({ price, y: yScale(price) });
    }
    return labels;
  }, [chartData, yScale]);

  // Generate time axis labels
  const timeLabels = useMemo(() => {
    if (candles.length === 0) return [];
    const labels = [];
    const step = Math.max(1, Math.floor(candles.length / 8));
    for (let i = 0; i < candles.length; i += step) {
      labels.push({
        timestamp: candles[i].timestamp,
        x: xScale(i),
        label: formatTime(candles[i].timestamp),
      });
    }
    return labels;
  }, [candles, xScale]);

  if (!chartData || candles.length === 0) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-64">
          <p className="text-zinc-500">No prediction data available</p>
        </div>
      </Card>
    );
  }

  const nowLineX = xScale(chartData.lastHistoricalIndex) + candleWidth;

  return (
    <Card className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            {symbol} - 90 Minute Prediction
          </h3>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            30 min historical + 60 min predicted
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Confidence badge */}
          <div className={`
            px-3 py-1 rounded-full text-sm font-medium
            ${confidence >= 0.7 ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' :
              confidence >= 0.5 ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400' :
              'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}
          `}>
            {(confidence * 100).toFixed(0)}% confidence
          </div>
          {/* Trend badge */}
          <div className={`
            px-3 py-1 rounded-full text-sm font-medium
            ${trend === 'bullish' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' :
              trend === 'bearish' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
              'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-400'}
          `}>
            {trend.charAt(0).toUpperCase() + trend.slice(1)}
          </div>
          {/* Refresh button */}
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors disabled:opacity-50"
            >
              {isLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto"
          style={{ maxHeight: '400px' }}
        >
          {/* Background */}
          <rect
            x={padding.left}
            y={padding.top}
            width={chartWidth}
            height={chartHeight}
            fill="transparent"
          />

          {/* Grid lines */}
          {priceLabels.map((label, i) => (
            <g key={i}>
              <line
                x1={padding.left}
                y1={label.y}
                x2={width - padding.right}
                y2={label.y}
                stroke={COLORS.gridLine}
                strokeWidth={0.5}
                strokeDasharray="4,4"
                opacity={0.3}
              />
              <text
                x={padding.left - 10}
                y={label.y + 4}
                textAnchor="end"
                fill={COLORS.text}
                fontSize={11}
              >
                ${formatPrice(label.price)}
              </text>
            </g>
          ))}

          {/* NOW divider line */}
          <line
            x1={nowLineX}
            y1={padding.top}
            x2={nowLineX}
            y2={height - padding.bottom}
            stroke={COLORS.nowLine}
            strokeWidth={2}
            strokeDasharray="6,4"
          />
          <text
            x={nowLineX}
            y={padding.top - 10}
            textAnchor="middle"
            fill={COLORS.nowLine}
            fontSize={12}
            fontWeight="bold"
          >
            NOW
          </text>

          {/* Section labels */}
          <text
            x={padding.left + (nowLineX - padding.left) / 2}
            y={padding.top - 10}
            textAnchor="middle"
            fill={COLORS.text}
            fontSize={11}
          >
            Historical (30 min)
          </text>
          <text
            x={nowLineX + (width - padding.right - nowLineX) / 2}
            y={padding.top - 10}
            textAnchor="middle"
            fill={COLORS.text}
            fontSize={11}
          >
            Predicted (60 min)
          </text>

          {/* Candles */}
          {candles.map((candle, index) => {
            const x = xScale(index) - candleWidth / 2;
            const color = getCandleColor(candle);
            const bodyTop = yScale(Math.max(candle.open, candle.close));
            const bodyBottom = yScale(Math.min(candle.open, candle.close));
            const bodyHeight = Math.max(1, bodyBottom - bodyTop);
            const wickX = x + candleWidth / 2;

            return (
              <g
                key={index}
                onMouseEnter={() => setHoveredCandle(candle)}
                onMouseLeave={() => setHoveredCandle(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* Wick (high-low line) */}
                <line
                  x1={wickX}
                  y1={yScale(candle.high)}
                  x2={wickX}
                  y2={yScale(candle.low)}
                  stroke={color}
                  strokeWidth={1}
                />
                {/* Body (open-close rectangle) */}
                <rect
                  x={x}
                  y={bodyTop}
                  width={candleWidth}
                  height={bodyHeight}
                  fill={candle.direction === 'up' ? color : color}
                  stroke={color}
                  strokeWidth={0.5}
                  opacity={candle.type === 'predicted' ? 0.85 : 1}
                />
              </g>
            );
          })}

          {/* Time axis labels */}
          {timeLabels.map((label, i) => (
            <text
              key={i}
              x={label.x}
              y={height - padding.bottom + 20}
              textAnchor="middle"
              fill={COLORS.text}
              fontSize={10}
            >
              {label.label}
            </text>
          ))}

          {/* Axis labels */}
          <text
            x={padding.left - 50}
            y={height / 2}
            textAnchor="middle"
            fill={COLORS.text}
            fontSize={12}
            transform={`rotate(-90, ${padding.left - 50}, ${height / 2})`}
          >
            Price ($)
          </text>
          <text
            x={width / 2}
            y={height - 10}
            textAnchor="middle"
            fill={COLORS.text}
            fontSize={12}
          >
            Time
          </text>
        </svg>

        {/* Tooltip */}
        {hoveredCandle && (
          <div className="absolute top-4 right-4 bg-zinc-900/95 dark:bg-zinc-800/95 text-white px-4 py-3 rounded-lg shadow-lg text-sm">
            <div className="font-medium mb-2">
              {formatTime(hoveredCandle.timestamp)}
              <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                hoveredCandle.type === 'historical' ? 'bg-zinc-700' : 'bg-purple-600'
              }`}>
                {hoveredCandle.type}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <span className="text-zinc-400">Open:</span>
              <span className="font-mono">${formatPrice(hoveredCandle.open)}</span>
              <span className="text-zinc-400">High:</span>
              <span className="font-mono">${formatPrice(hoveredCandle.high)}</span>
              <span className="text-zinc-400">Low:</span>
              <span className="font-mono">${formatPrice(hoveredCandle.low)}</span>
              <span className="text-zinc-400">Close:</span>
              <span className="font-mono">${formatPrice(hoveredCandle.close)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Legend and metadata */}
      <div className="flex flex-wrap items-center justify-between mt-4 pt-4 border-t border-zinc-200 dark:border-zinc-800">
        {/* Legend */}
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.historicalUp }} />
            <span className="text-zinc-600 dark:text-zinc-400">Historical Up</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.historicalDown }} />
            <span className="text-zinc-600 dark:text-zinc-400">Historical Down</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.predictedUp }} />
            <span className="text-zinc-600 dark:text-zinc-400">Predicted Up</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.predictedDown }} />
            <span className="text-zinc-600 dark:text-zinc-400">Predicted Down</span>
          </div>
        </div>

        {/* Metadata */}
        <div className="flex items-center gap-4 text-xs text-zinc-500 dark:text-zinc-400">
          <span>Generated {ageMinutes} min ago</span>
          <span>|</span>
          <span>Expires in {expiresInMinutes} min</span>
        </div>
      </div>
    </Card>
  );
});
