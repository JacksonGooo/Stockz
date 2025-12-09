'use client';

import { useMemo, useRef, useEffect, useState } from 'react';

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  isPrediction?: boolean;
}

interface CandlestickChartProps {
  realCandles: Candle[];
  predictedCandles: Candle[];
  width?: number;
  height?: number;
  autoFit?: boolean;
}

// Aggregate candles when there are too many to display clearly
function aggregateCandles(candles: Candle[], targetCount: number): Candle[] {
  if (candles.length <= targetCount) return candles;

  const ratio = Math.ceil(candles.length / targetCount);
  const aggregated: Candle[] = [];

  for (let i = 0; i < candles.length; i += ratio) {
    const chunk = candles.slice(i, Math.min(i + ratio, candles.length));
    if (chunk.length === 0) continue;

    aggregated.push({
      timestamp: chunk[0].timestamp,
      open: chunk[0].open,
      high: Math.max(...chunk.map(c => c.high)),
      low: Math.min(...chunk.map(c => c.low)),
      close: chunk[chunk.length - 1].close,
      volume: chunk.reduce((sum, c) => sum + c.volume, 0),
      isPrediction: chunk[0].isPrediction,
    });
  }

  return aggregated;
}

export function CandlestickChart({
  realCandles,
  predictedCandles,
  width: propWidth,
  height = 400,
  autoFit = true,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(propWidth || 900);

  // Auto-detect container width
  useEffect(() => {
    if (!autoFit || !containerRef.current) return;

    const updateWidth = () => {
      if (containerRef.current) {
        const newWidth = containerRef.current.offsetWidth;
        if (newWidth > 0) setContainerWidth(newWidth);
      }
    };

    updateWidth();
    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, [autoFit]);

  const width = autoFit ? containerWidth : (propWidth || 900);
  const allCandles = useMemo(() => {
    const real = realCandles.map(c => ({ ...c, isPrediction: false }));
    const predicted = predictedCandles.map(c => ({ ...c, isPrediction: true }));
    return [...real, ...predicted];
  }, [realCandles, predictedCandles]);

  const chartData = useMemo(() => {
    if (allCandles.length === 0) {
      return { minPrice: 0, maxPrice: 100, priceRange: 100, candles: [] };
    }

    // Find price range
    let minPrice = Infinity;
    let maxPrice = -Infinity;

    for (const candle of allCandles) {
      minPrice = Math.min(minPrice, candle.low);
      maxPrice = Math.max(maxPrice, candle.high);
    }

    // Add padding
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.1;
    minPrice -= padding;
    maxPrice += padding;

    return {
      minPrice,
      maxPrice,
      priceRange: maxPrice - minPrice,
      candles: allCandles,
    };
  }, [allCandles]);

  const padding = { top: 20, right: 60, bottom: 40, left: 20 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Convert price to Y coordinate
  const priceToY = (price: number) => {
    const { minPrice, priceRange } = chartData;
    return chartHeight - ((price - minPrice) / priceRange) * chartHeight + padding.top;
  };

  // Maximum candles to display (auto-aggregate if more)
  const maxDisplayCandles = Math.floor(chartWidth / 6); // Min 6px per candle
  const displayCandles = useMemo(() => {
    return aggregateCandles(allCandles, maxDisplayCandles);
  }, [allCandles, maxDisplayCandles]);

  const displayCandleWidth = Math.max(2, Math.min(12, chartWidth / displayCandles.length - 2));
  const displayCandleSpacing = displayCandles.length > 0 ? chartWidth / displayCandles.length : 0;

  // Find the divider position for display candles
  const realDisplayCount = displayCandles.filter(c => !c.isPrediction).length;
  const displayDividerX = realDisplayCount * displayCandleSpacing + padding.left - displayCandleSpacing / 2;

  // Generate price labels
  const priceLabels = useMemo(() => {
    const labels: { price: number; y: number; label: string }[] = [];
    const { minPrice, maxPrice } = chartData;
    const steps = 5;
    const stepSize = (maxPrice - minPrice) / steps;

    for (let i = 0; i <= steps; i++) {
      const price = minPrice + stepSize * i;
      labels.push({
        price,
        y: priceToY(price),
        label: price.toFixed(2),
      });
    }
    return labels;
  }, [chartData]);

  // Generate time labels
  const timeLabels = useMemo(() => {
    const labels: { x: number; label: string; isPrediction: boolean | undefined }[] = [];
    if (displayCandles.length === 0) return labels;

    // Show labels at intervals
    const interval = Math.max(1, Math.floor(displayCandles.length / 8));

    for (let i = 0; i < displayCandles.length; i += interval) {
      const candle = displayCandles[i];
      const date = new Date(candle.timestamp);
      const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      labels.push({
        x: i * displayCandleSpacing + padding.left + displayCandleWidth / 2,
        label: time,
        isPrediction: candle.isPrediction,
      });
    }
    return labels;
  }, [displayCandles, displayCandleSpacing, displayCandleWidth]);

  if (allCandles.length === 0) {
    return (
      <div
        ref={containerRef}
        className="flex items-center justify-center bg-zinc-900 rounded-lg border border-zinc-700 w-full"
        style={{ height }}
      >
        <p className="text-zinc-500">No data to display</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg
        width={width}
        height={height}
        className="bg-zinc-900 rounded-lg border border-zinc-700"
      >
        {/* Grid lines */}
        {priceLabels.map((label, i) => (
          <line
            key={i}
            x1={padding.left}
            x2={width - padding.right}
            y1={label.y}
            y2={label.y}
            stroke="#27272a"
            strokeWidth={1}
          />
        ))}

        {/* Divider line between real and predicted */}
        {predictedCandles.length > 0 && (
          <>
            <line
              x1={displayDividerX}
              x2={displayDividerX}
              y1={padding.top}
              y2={height - padding.bottom}
              stroke="#71717a"
              strokeWidth={2}
              strokeDasharray="6,4"
            />
            <text
              x={displayDividerX}
              y={padding.top - 5}
              fill="#a1a1aa"
              fontSize={10}
              textAnchor="middle"
            >
              NOW
            </text>
          </>
        )}

        {/* Candles */}
        {displayCandles.map((candle, i) => {
          const x = i * displayCandleSpacing + padding.left;
          const isUp = candle.close >= candle.open;
          const bodyTop = priceToY(Math.max(candle.open, candle.close));
          const bodyBottom = priceToY(Math.min(candle.open, candle.close));
          const bodyHeight = Math.max(1, bodyBottom - bodyTop);
          const wickTop = priceToY(candle.high);
          const wickBottom = priceToY(candle.low);

          // Colors: green for up, red for down
          // Predicted candles are semi-transparent
          const opacity = candle.isPrediction ? 0.5 : 1;
          const fillColor = isUp
            ? `rgba(34, 197, 94, ${opacity})`  // green
            : `rgba(239, 68, 68, ${opacity})`; // red
          const strokeColor = isUp
            ? `rgba(22, 163, 74, ${opacity})`  // darker green
            : `rgba(220, 38, 38, ${opacity})`; // darker red

          return (
            <g key={i}>
              {/* Wick */}
              <line
                x1={x + displayCandleWidth / 2}
                x2={x + displayCandleWidth / 2}
                y1={wickTop}
                y2={wickBottom}
                stroke={strokeColor}
                strokeWidth={1}
              />
              {/* Body */}
              <rect
                x={x}
                y={bodyTop}
                width={displayCandleWidth}
                height={bodyHeight}
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={1}
                rx={1}
              />
            </g>
          );
        })}

        {/* Price labels (right side) */}
        {priceLabels.map((label, i) => (
          <text
            key={i}
            x={width - padding.right + 5}
            y={label.y + 4}
            fill="#a1a1aa"
            fontSize={11}
            fontFamily="monospace"
          >
            ${label.label}
          </text>
        ))}

        {/* Time labels (bottom) */}
        {timeLabels.map((label, i) => (
          <text
            key={i}
            x={label.x}
            y={height - padding.bottom + 20}
            fill={label.isPrediction ? '#71717a' : '#a1a1aa'}
            fontSize={10}
            textAnchor="middle"
            fontFamily="monospace"
          >
            {label.label}
          </text>
        ))}

        {/* Legend */}
        <g transform={`translate(${padding.left + 10}, ${padding.top + 10})`}>
          <rect x={0} y={0} width={12} height={12} fill="rgba(34, 197, 94, 1)" rx={2} />
          <text x={18} y={10} fill="#a1a1aa" fontSize={11}>Real Data</text>

          <rect x={90} y={0} width={12} height={12} fill="rgba(34, 197, 94, 0.5)" rx={2} />
          <text x={108} y={10} fill="#a1a1aa" fontSize={11}>Predicted</text>
        </g>
      </svg>
    </div>
  );
}
